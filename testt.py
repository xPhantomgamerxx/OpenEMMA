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

model_path = "deepseek-ai/Janus-Pro-7B"
vlm_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vlm_chat_processor.tokenizer
vlm: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)

image = "/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402932162460.jpg"

prompt = [{"role": "User",
    "content": f"<image_placeholder>\n Describe the image in detail",
    "images": [image]},
    {"role": "Assistant", "content": ""}]

pil_images = load_pil_images(prompt)

prepare_inputs = vlm_chat_processor(conversations=prompt, images=pil_images, force_batchify=True).to(vlm.device)
inputs_embeds = vlm.prepare_inputs_embeds(**prepare_inputs)
outputs = vlm.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=4096,
    do_sample=False,
    use_cache=True)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).replace("\n\n", " ")

print(answer)
