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
# from truckscenes import TruckScenes
from transformers import AutoModelForCausalLM, pipeline
from openemma.YOLO3D.inference import yolo3d_nuScenes
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from Janus.janus.utils.io import load_pil_images


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
    prepare_inputs = chat_processor(conversations=message, images=pil_images, force_batchify=True).to(model.device)
    
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
    print(inputs_embeds.dtype)

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

model_path = "deepseek-ai/Janus-Pro-7B"
vlm_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vlm_chat_processor.tokenizer
vlm: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)

with open("/home/ubuntu/project_ws/OpenEMMA/lane_test/frame_00000011.json") as f:
    data = json.load(f)
annotated_img = "/home/ubuntu/project_ws/OpenEMMA/lane_test/frame_00000011_jpg_annotated_class.jpg"
raw_img = "/home/ubuntu/project_ws/OpenEMMA/lane_test/frame_00002540.jpg"

prompt = [{"role": "User",
    "content": f"<image_placeholder>\n describe this image",
    "images": [raw_img]},
    {"role": "Assistant", "content": ""}]

conversation = [
    {
        "role": "<|User|>",
        "content": "This is image_1: <image_placeholder>\n \
                    This is image_2: <image_placeholder>\n \
                    Find the lanesplitting lines in detail and ",
        "images": [annotated_img,raw_img],
    },
    {"role": "<|Assistant|>", "content": ""}
]
answer, _ = vlm_inference(prompt, vlm_chat_processor, vlm, verbose=True)
print(answer)

