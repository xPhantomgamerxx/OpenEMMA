from __future__ import annotations
import os
import cv2
import argparse
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from math import atan2
from nuscenes import NuScenes
from truckscenes import TruckScenes
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor, AutoProcessor, pipeline, AutoModel
from openemma.YOLO3D.inference import yolo3d_nuScenes
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from Janus.janus.utils.io import load_pil_images
from utils import EstimateCurvatureFromTrajectory, IntegrateCurvatureForPoints

from DeepSeek_VL2.deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from DeepSeek_VL2.deepseek_vl2.utils.io import load_pil_images




model_path = "deepseek-ai/Janus-Pro-1B"
model_path = "deepseek-ai/deepseek-vl2-small"
# vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
# if torch.cuda.is_available():
#     vl_gpt = vl_gpt.to(torch.bfloat16).cuda()

img_path = [
    "/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402932162460.jpg",
    "/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402932612460.jpg"]

message = "These images are taken from a car driving along the street at 0.5s time difference between them. what speed and steering angle would you infer from these 2 images, give me an answer in m/s and rad/s"
prompt = [{
    "role": "User",
    "content": f"<image_placeholder> is Figure1\n <image_placeholder> is Figure2\n{message}",
    "images": img_path,
    },
    {"role": "Assistant", "content": ""},]

prompt = [
    {
        "role": "<|User|>",
        "content": "<image>\n You are an autonomous driving labeller, give me all of the important information with regards to self driving in this scene.",
        "images": ["/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402932162460.jpg"],
    },
    {"role": "<|Assistant|>", "content": ""},
]

pil_images = load_pil_images(prompt)
prepare_inputs = vl_chat_processor(conversations=prompt, images=pil_images, force_batchify=True).to(vl_gpt.device)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# run the model to get the response
outputs = vl_gpt.language.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=2048,
    do_sample=False,
    use_cache=True)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).replace("\n\n", " ")
print(answer)
full_answer = (f"{prepare_inputs['sft_format'][0]}", answer)