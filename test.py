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



#tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
#model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
# processor = AutoProcessor.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
#image_processor = AutoImageProcessor.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")


nusc = NuScenes(version='v1.0-mini', dataroot='/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes')
nusc = TruckScenes(version='v1.0-mini', dataroot='/home/ubuntu/project_ws/OpenEMMA/datasets/truck_scenes/truck_scenes/man-truckscenes')
scene = nusc.scene[0]
token = scene['token']
curr_sample_token = scene['first_sample_token']
last_sample_token = scene['last_sample_token']
name = scene['name']
description = scene['description']

sample = nusc.get('sample', curr_sample_token)
#cam_front_img = nusc.get('sample_data', sample['data']['CAM_FRONT'])
cam_front_img = nusc.get('sample_data', sample['data']['CAMERA_LEFT_FRONT'])
img_path = os.path.join(nusc.dataroot, cam_front_img['filename'])
camera_params = nusc.get('calibrated_sensor', cam_front_img['calibrated_sensor_token'])
with open(os.path.join(img_path), "rb") as image_file:
    img = cv2.imdecode(np.frombuffer(image_file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    #img = yolo3d_nuScenes(img, calib=camera_params)[0]
    #cv2.imwrite("testimg.jpg", img)

# messages = [{"role": "user", "content": "How do I implement a query to your model when loading the files directly through transformer?"}]
text1 = "Hello"
text2 = " who are you?"
message = [{"role": "user","content": text1+text2}]
# encoded_msg = tokenizer(message)
# print(encoded_msg)


pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device=0, max_new_tokens=2048)
msg = pipe(message)
print(msg)


model_path = "deepseek-ai/Janus-Pro-1B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
if torch.cuda.is_available():
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda()

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

pil_images = load_pil_images(prompt)
prepare_inputs = vl_chat_processor(conversations=prompt, images=pil_images, force_batchify=True).to(vl_gpt.device)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# run the model to get the response
outputs = vl_gpt.language_model.generate(
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