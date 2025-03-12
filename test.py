# pip install 'git+https://github.com/huggingface/transformers.git@v4.49.0-AyaVision'
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import cv2
import os
import numpy as np
import time

model_id = "CohereForAI/aya-vision-8b"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.float16
)

with open(os.path.join("/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402932162460.jpg"), "rb") as f:
    img = cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    # Format message with the aya-vision chat template
messages = [
    {"role": "user", 
     "content": [
       {"type": "image", "url": "/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402932162460.jpg"},
        {"type": "text", "text": "You are an autonomous driving labeller. You have access to this front-view camera image taken from a driving car. Imagine you are driving the car. Based on what you see, give me three things. 1. A scene description, where you describe the driving scene with regards to all aspects you think are important for driving safely. 2. An object description, were you describe the other road users that you are observing to ensure the safety of everyone in this scenario. 3. An intent description, based on the lane markings and the other road users you have detected, describe the best course of action you as a driver can take in the scenario."},
    ]},
    ]

inputs = processor.apply_chat_template(
    messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)
tic = time.time()
gen_tokens = model.generate(
    **inputs, 
    max_new_tokens=1000, 
    do_sample=True, 
    temperature=0.3,
)
toc = time.time()
print(f"Generation took {toc - tic:.2f} s")
print(processor.tokenizer.decode(gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
