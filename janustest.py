import ray
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import cv2
import os
import numpy as np
import time
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from Janus.janus.utils.io import load_pil_images
from transformers import AutoModelForCausalLM, pipeline


@ray.remote(num_gpus=1)
class JanusModelServer:
    def __init__(self, model_id: str):
        self.processor = VLChatProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
        self.model.eval()

    def run_inference(self, message: list, max_new_tokens: int = 1024):
        pil_images = load_pil_images(message)
        prepare_inputs = self.processor(conversations=message, images=pil_images, force_batchify=True).to(self.model.device) 

        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        outputs = self.model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=self.tokenizer.eos_token_id,
        bos_token_id=self.tokenizer.bos_token_id,
        eos_token_id=self.tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True)

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).replace("\n\n", " ")
        return answer
    
if __name__ == "__main__":
    ray.init()
    model_id = "deepseek-ai/Janus-Pro-7B"
    img_path = (
        "/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes/samples/"
        "CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402932162460.jpg"
    )

    with open(img_path, "rb") as f:
        img = cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR)

    messages = [{
            "role": "User",
            "content": "<image_placeholder>\nYou are an autonomous driving labeller. You have access to this front-view camera image taken from a driving car. Imagine you are driving the car. Reason about what you see in the image fully and based on what you see, provide me with these 3 things  . 1. A scene description, where you describe the driving scene with regards to all aspects you think are important for driving safely. 2. An object description, were you describe the other road users that you are observing to ensure the safety of everyone in this scenario. 3. An intent description, based on the lane markings and the other road users you have detected, describe the best course of action you as a driver can take in the scenario.",
            "images": [img_path],
            },
            {"role": "Assistant", "content": ""},
        ]
    
    num_actors = 4
    servers = [JanusModelServer.remote(model_id) for _ in range(num_actors)]
    start_time = time.time()
    inference_futures = [
        server.run_inference.remote(messages)
        for server in servers
    ]
    # Wait for all results
    results = ray.get(inference_futures)

    end_time = time.time()

    # 5. Print total elapsed time and each model's output
    print(f"Total parallel inference time: {end_time - start_time:.2f} seconds")
    for i, output in enumerate(results):
        print(f"==== Output from model {i} ====")
        print(output, "\n")
