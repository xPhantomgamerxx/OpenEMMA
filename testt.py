from transformers import pipeline
import torch


pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", max_new_tokens=4096, device_map="auto")
with open("prompt.txt", 'r') as f:
    message = f.read()
print("prompt loaded")
msg = pipe(message)
print(msg)

