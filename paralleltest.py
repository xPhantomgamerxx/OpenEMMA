# Make sure you install your custom Transformers first (the AyaVision branch):
# pip install 'git+https://github.com/huggingface/transformers.git@v4.49.0-AyaVision'

import ray
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import cv2
import os
import numpy as np
import time

# ------------------------------------------------------------
# Ray Actor Definition
# ------------------------------------------------------------
@ray.remote(num_gpus=1)
class AyaVisionModelServer:
    def __init__(self, model_id: str):
        # Load model + processor. 
        # Since Ray gives each actor exactly one GPU (num_gpus=1),
        # huggingface's `device_map="auto"` will see only one device
        # and place the model on that GPU automatically.
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
        # Optional: put model into eval mode
        self.model.eval()

    def run_inference(self, image_path: str, messages: list, max_new_tokens: int = 1000):
        # 1. Read the image
        with open(image_path, "rb") as f:
            img_data = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        
        # 2. Prepare the inputs via the AyaVision chat template
        inputs = self.processor.apply_chat_template(
            messages,
            padding=True,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        # 3. Generate
        gen_tokens = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=0.3,
        )

        # 4. Decode
        answer = self.processor.tokenizer.decode(
            gen_tokens[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        return answer

# ------------------------------------------------------------
# Main: Create Ray actors & run parallel inference
# ------------------------------------------------------------
if __name__ == "__main__":
    # 1. Initialize Ray
    ray.init()

    # 2. Define your model ID & input
    model_id = "CohereForAI/aya-vision-8b"
    image_path = (
        "/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes/samples/"
        "CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402932162460.jpg"
    )

    # Example chat prompt
    messages = [
      {"role": "user",
        "content": [
          {"type": "image", 
            "url": image_path
          },
          {"type": "text", 
            "text": "You are an autonomous driving labeller. You have access to this front-view camera image taken from a driving car. Imagine you are driving the car. Based on what you see, give me three things. 1. A scene description, where you describe the driving scene with regards to all aspects you think are important for driving safely. 2. An object description, were you describe the other road users that you are observing to ensure the safety of everyone in this scenario. 3. An intent description, based on the lane markings and the other road users you have detected, describe the best course of action you as a driver can take in the scenario."
          }
        ]
      }
    ]

    # 3. Create 4 actors (each on its own GPU)
    #    Adjust the number (range(4)) as desired for however many GPUs you have.
    num_actors = 4
    servers = [AyaVisionModelServer.remote(model_id) for _ in range(num_actors)]

    # 4. Trigger inference on each actor IN PARALLEL
    start_time = time.time()
    
    # Send all requests asynchronously
    inference_futures = [
        server.run_inference.remote(image_path, messages)
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
