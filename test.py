from __future__ import annotations

import os
import cv2
import re
import argparse
import torch
import logging
import json
import pytz
import numpy as np
import matplotlib.pyplot as plt

from math import atan2
from datetime import datetime
from nuscenes import NuScenes
from truckscenes import TruckScenes
from transformers import AutoModelForCausalLM, pipeline
from openemma.YOLO3D.inference import yolo3d_nuScenes
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from Janus.janus.utils.io import load_pil_images
from utils import EstimateCurvatureFromTrajectory, IntegrateCurvatureForPoints, OverlayTrajectory, WriteImageSequenceToVideo


def vlm_inference(
    message:list[dict] = None, 
    chat_processor: VLChatProcessor = None, 
    model: MultiModalityCausalLM = None,
    verbose: bool = False
) -> tuple[str, str]:
    """ Runs inference on the provided model and returns the response from the VLM

    Args:
        message (list[dict]): The message that should be passed to the MLLM, in form of a dictionary with roles, content and images
        chat_processor (VLChatProcessor): The VLM chat processor to tokenize the input for the VLM
        model (MultiModalityCausalLM): VLM model to process the query and generate the response
        verbose (bool): Enables print statements

    Returns:
        answer (tuple[str,str]): The answer of the VLM along with the full answer including the input
    """

    pil_images = load_pil_images(message)
    prepare_inputs = chat_processor(conversations=message, images=pil_images, force_batchify=True).to(vl_gpt.device)
    
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=4096,
        do_sample=False,
        use_cache=True)

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).replace("\n\n", " ")
    
    full_answer = (f"{prepare_inputs['sft_format'][0]}", answer)
    if verbose:
        print("answer: \n", answer)
        print("full_answer \n", full_answer)
    return (answer, full_answer)


if __name__ == "__main__":
    model_path = "deepseek-ai/Janus-Pro-7B"
    vlm_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vlm_chat_processor.tokenizer
    vlm: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
    img = "/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402932162460.jpg"

    prompt = [{
        "role": "User",
        "content": f"<image_placeholder>\n You are an autonomous driving labeller. You have access to this front-view camera image of a car. Imagine you are driving the car and describe the driving scene according to all aspects you think are important for driving safety. This could include traffic lights, movement of other cars or pedestrians, and lane markings. Do not describe the movement of the ego vehicle.",
        "images": img},
        {"role": "Assistant", "content": ""},
    ]
    scene, _ = vlm_inference(prompt, vlm_chat_processor, vlm)

    prompt = [{
        "role": "User",
        "content": f"<image_placeholder>\n You are a autonomous driving labeller. You have access to this front-view camera image taken from a driving car. Imagine you are the driver of the car. What other road users are you paying attention to in the driving scene? List two or three of them, specifying the location within the image of the driving scene and provide a short description of what that road user is currently doing, what they might do in the future, and why it is important to you. Dont try to describe the movement of the ego vehicle",
        "images": img},
        {"role": "Assistant", "content": ""},
    ] 
    object, _ = vlm_inference(prompt, vlm_chat_processor, vlm)

    prompt = [{
        "role": "User",
        "content": f"<image_placeholder>\n You are a autonomous driving labeller. You have access to this front-view camera image taken from a driving vehicle. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, describe the best course of action for the current car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?",
        "images": img},
        {"role": "Assistant", "content": ""},
    ]
    intent, _ = vlm_inference(prompt, vlm_chat_processor, vlm)

    prompt = [{
        "role": "User",
        "content": f"You are a driving expert driving a car in a real world scenario. 
    The scene is described as follows: {scene}. 
    The identified critical objects are {object}. 
    The current intent is {intent}. 
    The 5 second historical velocities and curvatures of the ego car are . 
    Output ONLY your predictions for the future speeds and curvatures of the vehicle in the style of [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10] for the next 10 timesteps in the style of a python tuple using square brackets. If the otuput doesn't meet the specifications it will be invalid, if there is ambiguity, assume the 5 seconds of historical velocities are correct",
        "images": img,
        },
        {"role": "Assistant", "content": ""},
    ]