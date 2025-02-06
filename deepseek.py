from __future__ import annotations
import os
import cv2
import argparse
import numpy as np
import torch
import logging
from nuscenes import NuScenes
from truckscenes import TruckScenes
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor, AutoProcessor, pipeline, AutoModel
from openemma.YOLO3D.inference import yolo3d_nuScenes
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from Janus.janus.utils.io import load_pil_images

logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torchvisionc').setLevel(logging.ERROR)


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
        answer (str): The answer of the VLM
        full_answer (str): The answer along with the input to the VLM

    """

    pil_images = load_pil_images(message)
    prepare_inputs = chat_processor(conversations=message, images=pil_images, force_batchify=True).to(vl_gpt.device)
    
    # run image encoder to get the image embeddings
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True)

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    full_answer = (f"{prepare_inputs['sft_format'][0]}", answer)
    if verbose:
        print("answer: \n", answer)
        print("full_answer \n", full_answer)
    return (answer, full_answer)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="janus1")
    parser.add_argument("--dataroot", type=str, default="/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes")
    parser.add_argument("--version", type=str, default='v1.0-mini')
    parser.add_argument("--vehicle", type=str, default='car')
    args = parser.parse_args()
    if args.model == "janus1":
        model_path = "deepseek-ai/Janus-Pro-1B"
    elif args.model== "janus7":
        model_path = "deepseek-ai/Janus-Pro-7B"

    # max_memory = {0:f"{(torch.cuda.get_device_properties('cuda:0').total_memory//(1024 ** 3)) * 0.8}GiB"}
    
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    if torch.cuda.is_available():
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda()
    
    img_path = "/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes/sweeps/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402932012460.jpg"
    
    with open(os.path.join(img_path), "rb") as img_file:
        img = cv2.imdecode(np.frombuffer(img_file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite("testimg.jpg", img)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot)
    scenes = nusc.scene
    scene_list = ["scene-0061"]#["scene-0103", "scene-1077"]

    conversation = [{
        "role": "User",
        "content": "<image_placeholder>\nYou are an advanced autonomous driving labeller, viewing the scene from a driver's perspective. Carefully analyze the input image and describe every detail relevant to driving safely. If available, include information about the road layout, lane markings, traffic signs, traffic signals, nearby vehicles, pedestrians, cyclists, environmental conditions (lighting, weather, road surface), potential obstacles, and any other noteworthy elements that could impact driving decisions. Your description should be comprehensive and precise, focusing on the aspects necessary for an autonomous vehicle to understand and navigate the environment reliably. Present your observations in a way that reflects how a self-driving car would perceive and label each element in the scene.",
        "images": [img_path],
        },
        {"role": "Assistant", "content": ""},
    ]
    answer, full_answer = vlm_inference(conversation, vl_chat_processor, vl_gpt)
    # print(f"Scene Description: \n{answer}")

    conversation = [{
        "role": "User",
        "content": "<image_placeholder>\nYou are a autonomous driving labeller. Imagine you are driving the car. You need to detect all of the objects that are in the image thatyou need to take into account to drive safely through the scene. List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you.",
        "images": [img_path],
        },
        {"role": "Assistant", "content": ""},
    ]
    
    answer, full_answer = vlm_inference(conversation, vl_chat_processor, vl_gpt)
    print(f"Object Detection: \n{answer}")


    # print(answer)