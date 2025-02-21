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
    prepare_inputs = chat_processor(conversations=message, images=pil_images, force_batchify=True).to(model.device)
    
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

# model_path = "deepseek-ai/Janus-Pro-7B"
# vlm_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
# tokenizer = vlm_chat_processor.tokenizer
# vlm: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)

deepseek_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
llm_pipe = pipeline("text-generation", model=deepseek_model, device_map="auto", max_new_tokens=4096)  

img = "/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402932162460.jpg"

prompt = [{"role": "User",
    "content": f"<image_placeholder>\n You are driving a car, you have access to this image from a front facing camera. Describe this image in detail, specifically with regards to autonomous driving. Then give me an estimate of what the ego vehicle that you are driving will do over the next 5 seconds, be as detailed as you can be, imagine the car moving forwards in the image continuing what you outline it to do. In the end give a brief description of what you conclude. What direction does the arrow infront of the vehicle point to and what does that mean for the future trajectory of the ego vehicle",
    "images": [img]},
    # {"role": "System_Message",
    # "content": """You are an AI assistant that must only follow a predefined output format. Strictly adhere to this format for all responses:
    # [Full reasoning process about how you get to the answer]
    # Final Answer:
    # 1. Object1
    # 2. Object2
    # 3. ...
    # """},
    {"role": "Assistant", "content": ""}]

prompt = [{"role": "User",
    "content": f"<image_placeholder>\n There is an arrow depicted on the street in this image, can you tell me what direction the arrow is pointing from the perspective of the camera. This camera is mounted to a car, what direction wil the car turn at the upcoming intersection, describe the actions over the next 5 seconds in detail",
    "images": [img]},

    {"role": "Assistant", "content": ""}]

# answer, _ = vlm_inference(prompt, vlm_chat_processor, vlm)
# print(answer)
# with open(os.path.join(img), "rb") as image_file:
#     img = cv2.imdecode(np.frombuffer(image_file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
# cv2.imwrite(f"testtt.jpg", img)
# prompt =  """
# [System]
# You are an AI assistant that must strictly follow a predefined output format.
# The outputformat is that of python list with each entry being an integer, ignore any delimiters like ',' or '.'
# Strictly adhere to this format in all responses:

# [Full reasoning process]
# Final Answer:
# [int, int, ..., int]

# [User]
# write the first 10 digits of pi

# [Assistant]
# """

answer = llm_pipe(prompt)
print(answer)