from __future__ import annotations

import os
import cv2
import re
import argparse
import torch
import logging
import json
import pytz
import ray
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

# logging.getLogger('transformers').setLevel(logging.ERROR)
# logging.getLogger('torchvisionc').setLevel(logging.ERROR)


@ray.remote(num_gpus=1)
class JanusModelServer:
    def __init__(self, model_id: str):
        self.processor = VLChatProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
        self.model.eval()
        self.verbose = False

    def vlm_inference(
            self,
            img: str = None,
            task: str = None,
            description: str = None,
            past_speed_curvature_str: str = None,
            max_new_tokens: int = 1024):
        """ Runs inference on the provided model and returns the response from the VLM

        Args:
            message (list[dict]): The message that will be passed to the VLM
            chat_processor (VLChatProcessor): The chat processor to tokenize the input for the VLM
            model (MultiModalityCausalLM): VLM model to process the query and generate the response
            verbose (bool): Enables print statements

        Returns:
            answer (str): The detokenized answer of the VLM 
        """

        if task == "description":
            prompt = [
                {"role": "User",
                 "content": "<image_placeholder>\nYou are an autonomous driving labeller. You have access to this front-view camera image taken from a driving car. Imagine you are driving the car. Reason about what you see in the image fully and based on what you see, provide me with these 3 things  . 1. A scene description, where you describe the driving scene with regards to all aspects you think are important for driving safely. 2. An object description, were you describe the other road users that you are observing to ensure the safety of everyone in this scenario. 3. An intent description, based on the lane markings and the other road users you have detected, describe the best course of action you as a driver can take in the scenario.",
                 "images": [img],},
                {"role": "Assistant", "content": ""},]
        elif task == "final":
            pattern = (
                r"1\.\s*\*\*Scene Description:\*\*\s*(.*?)\s*"
                r"2\.\s*\*\*Object Description:\*\*\s*(.*?)\s*"
                r"3\.\s*\*\*Intent Description:\*\*\s*(.*)")

            match = re.search(pattern, description, re.DOTALL)
            if match:
                scene_description = match.group(1).strip()
                object_description = match.group(2).strip()
                intent_description = match.group(3).strip()
            else:
                print("No match found.")
            prompt = [
                {"role": "User",
                 "content": f"<image_placeholder>\nYou are a driving expert in this scenario. The scene you must analyze is described by: {scene_description} The most important objects have been described as: {object_description} The current intent of the vehicle is described as: {intent_description} The historical velocities and curvatures of the ego car of the last 5 seconds up until the present are: {past_speed_curvature_str}. You must reason about the scene fully, then make a prediction about the next 10 velocities and curvatures the vehicle shall take. Provide these in the format of [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10] in the style of a python tuple. f there is ambiguity, assume the 5 seconds of historical velocities are correct. The predicted speed and curvature should continue from where the past values left off.",
                 "images": [img],},
                {"role": "Assistant", "content": ""},]
        
        pil_images = load_pil_images(prompt)
        prepare_inputs = self.processor(conversations=prompt, images=pil_images, force_batchify=True).to(self.model.device)
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
        if self.verbose:
            full_answer = (f"{prepare_inputs['sft_format'][0]}", answer)
            print("answer: \n", answer)
            print("full_answer \n", full_answer)
        return answer
    
    def generate_motion(
        self,
        current_image: str = None,
        past_velocities = None,
        past_curvatures = None):
        """Generates the motion of the vehicle based on the past velocities, curvatures, and intent"""

        descriptions = self.vlm_inference(current_image)

        past_velocities_norm = np.linalg.norm(past_velocities, axis=1)
        past_curvatures = past_curvatures * 100
        past_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(past_velocities_norm, past_curvatures)]
        past_speed_curvature_str = ", ".join(past_speed_curvature_str)

        final = self.vlm_inference(
            img=current_image,
            task="final",
            description=descriptions,
            past_speed_curvature_str=past_speed_curvature_str)
        
        return final
    
if __name__ == "__main__":
    ray.init()
    model_id = "deepseek-ai/Janus-Pro-7B"
    parser = argparse.ArgumentParser(description="Run inference on the Janus model")
    parser.add_argument("--dataroot", type=str, default="/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    local_tz = pytz.timezone("Europe/Stockholm")
    timestamp = datetime.now(local_tz).strftime("%m-%d_%H-%M")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot)
    scenes = nusc.scene
    scene_list = ["scene-0061"]

    for scene in scenes:
        tic = datetime.now()
        name = scene['name'] 
        if name not in scene_list:
            continue
        token = scene['token']
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        description = scene['description']
        path = f"car_results/deepseek/{name}/{timestamp}"
        os.makedirs(f"{path}", exist_ok = True)
        front_cam_images = []
        ego_poses = []
        camera_params = []
        current_sample_token = first_sample_token

        while True:
            current_sample = nusc.get('sample', current_sample_token)
            cam_front_data = nusc.get('sample_data', current_sample['data']['CAM_FRONT'])
            front_cam_images.append(os.path.join(nusc.dataroot, cam_front_data['filename']))
            camera_params.append(nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token']))
            pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
            ego_poses.append(pose)
            if current_sample_token == last_sample_token:
                break
            current_sample_token = current_sample['next']

        scene_length = len(front_cam_images)
        print(f"Scene {name} loaded properly with {scene_length} frames")

        ego_poses_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]
        ego_poses_world = np.array(ego_poses_world)
        # Get the velocities of the ego vehicle.
        ego_velocities = np.zeros_like(ego_poses_world)
        ego_velocities[1:] = ego_poses_world[1:] - ego_poses_world[:-1]
        ego_velocities[0] = ego_velocities[1]
        # Get the curvature of the ego vehicle and predict the points based on the velocity and curvature
        ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world)
        ego_velocities_norm = np.linalg.norm(ego_velocities, axis=1)
        estimated_points = IntegrateCurvatureForPoints(ego_curvatures, ego_velocities_norm, ego_poses_world[0],atan2(ego_velocities[0][1], ego_velocities[0][0]), scene_length)
        # Trajectory of the ego vehicle in the world pose
        ego_traj_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]
        PAST_LENGTH = 10
        FUTURE_LENGHT = 10
        TOTAL_LENGTH = PAST_LENGTH + FUTURE_LENGHT
        USABLE_LENGTH = scene_length - FUTURE_LENGHT
        cam_images_sequence = []
        ade1s_list = []
        ade2s_list = []
        ade3s_list = []
        prev_intent = []

        for i in range(scene_length-FUTURE_LENGHT):
            past_images = front_cam_images[i:i+PAST_LENGTH]
            past_ego_poses = ego_poses[i:i+PAST_LENGTH]
            past_camera_params = camera_params[i:i+PAST_LENGTH]
            past_ego_traj_world = ego_traj_world[i:i+PAST_LENGTH]