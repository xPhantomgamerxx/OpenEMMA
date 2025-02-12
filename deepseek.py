from __future__ import annotations
import os
import cv2
import re
import argparse
import numpy as np
import torch
import logging
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
        answer (tuple[str,str]): The answer of the VLM along with the full answer including the input
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
        max_new_tokens=4096,
        do_sample=False,
        use_cache=True)

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).replace("\n\n", " ")
    full_answer = (f"{prepare_inputs['sft_format'][0]}", answer)
    if verbose:
        print("answer: \n", answer)
        print("full_answer \n", full_answer)
    return (answer, full_answer)

def call_vlm(
    message: list[dict] = None,
    img: str = None,
    chat_processor: VLChatProcessor = None,
    model: MultiModalityCausalLM = None,
    task: str = None,
    verbose: bool = False
) -> str:
    """ Calls the VLM with the task specific prompt
    
    Args:
        message (list[dict]): prompt to describe the scene 
        img_path (str): path to the img file that should be described
        chat_processor (VLChatProcessor): Texxt tokenizer
        model (MultiMidalityCausalLM): VLM model that will be prompted
        task (str): What task is being addressed
        verbose (bool): Enables printing
        
    Returns:
        answer (str): answer of the model
    """
    if task == None:
        prompt = [{
            "role": "User",
            "content": "<image_placeholder>\nYou are an advanced autonomous driving labeller, with access to a front-view camera image of a vehicle. Carefully analyze the input image and describe every detail relevant to driving safely. If available, include information about the road layout, lane markings, traffic signs, traffic signals, nearby vehicles, pedestrians, cyclists, environmental conditions (lighting, weather, road surface), potential obstacles, and any other noteworthy elements that could impact driving decisions. Your description should be comprehensive and precise, focusing on the aspects necessary for an autonomous vehicle to understand and navigate the environment reliably. Present your observations in a way that reflects how a self-driving car would perceive and label each element in the scene.",
            "images": img,
            },
            {"role": "Assistant", "content": ""},
        ]
    elif task =="scene":
        prompt = [{
            "role": "User",
            "content": f"<image_placeholder>\n You are an autonomous driving labeller. You have access to this front-view camera image of a car. Imagine you are driving the car and describe the driving scene according to all aspects you think are important for driving safety. This could include traffic lights, movement of other cars or pedestrians, and lane markings. Do not describe the movement of the ego vehicle.",
            "images": img,
            },
            {"role": "Assistant", "content": ""},
        ]
    elif task == "object":
        prompt = [{
            "role": "User",
            "content": f"<image_placeholder>\n You are a autonomous driving labeller. You have access to this front-view camera image taken from a driving car. Imagine you are the driver of the car. What other road users are you paying attention to in the driving scene? List two or three of them, specifying the location within the image of the driving scene and provide a short description of what that road user is currently doing, what they might do in the future, and why it is important to you. Dont try to describe the movement of the ego vehicle",
            "images": img,
            },
            {"role": "Assistant", "content": ""},
        ] 
    elif task == "intent":
        if message == None:
            prompt = [{
                "role": "User",
                "content": f"<image_placeholder>\n You are a autonomous driving labeller. You have access to this front-view camera image taken from a driving vehicle. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, describe the best course of action for the current car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?",
                "images": img,
                },
                {"role": "Assistant", "content": ""},
            ]
        else:
            prompt = [{
                "role": "User",
                "content": f"<image_placeholder>\n You are a autonomous driving labeller. You have access to this front-view camera image taken from a driving vehicle. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians given as: {message}, describe the best course of action for the current car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?",
                "images": img,
                },
                {"role": "Assistant", "content": ""},
            ]
    elif task == "final":
        prompt = [{
            "role": "User", 
            "content": f"<image_placeholder>\n {message}",
            "images": img,
            },
            {"role": "Assistant", "content": ""},
        ]

    answer, full_answer = vlm_inference(prompt, chat_processor, model)
    if verbose: 
        print("answer: \n", answer)
        print("full_answer \n", full_answer)
    return answer


def call_llm(
    message: (str) = None,
    llm_pipe: (pipeline) = None
) -> str:
    """Calls the LLM with the given prompt and returns the answer
    Args:
        message (str): The prompt to pass to the LLM (DeepSeek-R1-Distill-Qwen)
        llm_pipe (pipeline): The pipeline object that contains the LLM
    Returns:
        answer (str): The LLM's response to the prompt
    """
    prompt = [{"role": "user", "content": f"{message}"}]
    # with open("prompt.txt", 'w') as f:
    #    f.write(f"{prompt}")
    answer = llm_pipe(prompt)
    return answer
    
def GenerateMotion(
    current_image: str = None, 
    past_waypoints = None, 
    past_velocities = None, 
    past_curvatures = None, 
    given_intent = None, 
    chat_processor: VLChatProcessor = None,
    model: MultiModalityCausalLM = None,
    llm_pipe: pipeline = None,
    verbose: bool = None
) -> str:
    """Applies the OpenEMMA method of generating the reasoning process behind the prediction.
    
    Args:
        current_image (str): current image
        
    Returns:
        str
    """
    scene_description = call_vlm(message=None, img=current_image, chat_processor=chat_processor, model=model, task="scene")
    print("Scene Description done")
    if verbose: print(f"{scene_description}")
    object_description = call_vlm(message=None, img=current_image, chat_processor=chat_processor, model=model, task="object")
    print("Object Description done")
    if verbose: print(f"{object_description}")
    intent_description = call_vlm(message=object_description, img=current_image, chat_processor=chat_processor, model=model, task="intent")
    print("Intent Description done")
    if verbose: print(f"{intent_description}")
    
    past_waypoints_str = [f"[{x[0]:.2f},{x[1]:.2f}]" for x in past_waypoints]
    past_waypoints_str = ", ".join(past_waypoints_str)
    past_velocities_norm = np.linalg.norm(past_velocities, axis=1)
    past_curvatures = past_curvatures * 100
    past_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(past_velocities_norm, past_curvatures)]
    past_speed_curvature_str = ", ".join(past_speed_curvature_str)
    
    message = f"""You are a driving expert driving a car in a real world scenario. 
    The scene is described as follows: {scene_description}. 
    The identified critical objects are {object_description}. 
    The current intent is {intent_description}. 
    The 5 second historical velocities and curvatures of the ego car are {past_speed_curvature_str}. 
    Output ONLY your predictions for the future speeds and curvatures of the vehicle in the style of [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10] for the next 10 timesteps in the style of a python tuple. If the otuput doesn't meet the specifications it will be invalid, if there is ambiguity, assume the 5 seconds of historical velocities are correct"""
    if llm_pipe == None:
        print("Prompting VLM with full message")
        if verbose: print(f"Message that will be passed to VLM: \n{message}")
        final = call_vlm(message=message, img=current_image, chat_processor=chat_processor, model=model, task="final")
    else:
        print("Prompting LLM with full message")
        if verbose: print(f"Message that will be passed to LLM: \n{message}")
        final = call_llm(message=message, llm_pipe=llm_pipe)

    return final, scene_description, object_description, intent_description


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="janus7")
    parser.add_argument("--dataroot", type=str, default="/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes")
    parser.add_argument("--version", type=str, default='v1.0-mini')
    parser.add_argument("--vehicle", type=str, default='car')
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--plot", type=bool, default=True)
    args = parser.parse_args()
    if args.model == "janus1":
        model_path = "deepseek-ai/Janus-Pro-1B"
    elif args.model== "janus7":
        model_path = "deepseek-ai/Janus-Pro-7B"
    
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)

    llm_pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", device_map="auto", max_new_tokens=4096)  
    nusc = NuScenes(version=args.version, dataroot=args.dataroot)
    scenes = nusc.scene
    scene_list = ["scene-0103"]#["scene-0061", "scene-0103", "scene-1077"]
    timestamp = datetime.now().strftime("%m%d-%H%M")

    for scene in scenes:
        name = scene['name']
        if not name in scene_list:
            continue

        # Load all of the useful token info
        token = scene['token']
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        description = scene['description']
        os.makedirs(f"car_results/deepseek/{timestamp}/{name}", exist_ok = True)

        front_cam_images = []
        ego_poses = []
        camera_params = []
        current_sample_token = first_sample_token
        while True:
            # Load current sample and its data, append filepath to front_cam_images array + get camera params for the image
            current_sample = nusc.get('sample', current_sample_token)
            cam_front_data = nusc.get('sample_data', current_sample['data']['CAM_FRONT'])
            front_cam_images.append(os.path.join(nusc.dataroot, cam_front_data['filename']))
            camera_params.append(nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token']))


            # Same for the poses of the vehicle
            pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
            ego_poses.append(pose)

            if current_sample_token == last_sample_token:
                break
            current_sample_token = current_sample['next']

        scene_length = len(front_cam_images)
        print(f"Scene {name} loaded properly with {scene_length} frames")

        # Compute interpolated trajectory.
        ego_poses_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]
        ego_poses_world = np.array(ego_poses_world)
        plt.plot(ego_poses_world[:, 0], ego_poses_world[:, 1], 'r-', label='GT')
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

        prev_intent = None
        cam_images_sequence = []
        ade1s_list = []
        ade2s_list = []
        ade3s_list = []

        for i in range(scene_length - TOTAL_LENGTH):
            past_images = front_cam_images[i:i+PAST_LENGTH]
            past_ego_poses = ego_poses[i:i+PAST_LENGTH]
            past_camera_params = camera_params[i:i+PAST_LENGTH]
            past_ego_traj_world = ego_traj_world[i:i+PAST_LENGTH]
            future_ego_traj_world = ego_traj_world[i+PAST_LENGTH:i+TOTAL_LENGTH]
            past_ego_velocities = ego_velocities[i:i+PAST_LENGTH]
            past_ego_curvatures = ego_curvatures[i:i+PAST_LENGTH]

            past_start_world = past_ego_traj_world[0]
            future_start_world = past_ego_traj_world[-1]
            current_image = past_images[-1]
            with open(os.path.join(current_image), "rb") as image_file:
                img = cv2.imdecode(np.frombuffer(image_file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
            
            prediction, scene_description, object_description, intent_description = GenerateMotion(current_image=[current_image], past_waypoints=past_ego_traj_world, past_velocities=past_ego_velocities, past_curvatures=past_ego_curvatures, given_intent=prev_intent, chat_processor=vl_chat_processor, model=vl_gpt, llm_pipe=llm_pipe, verbose=args.verbose)

            with open(f"car_results/deepseek/{timestamp}/{name}/prediction_{i}.txt", 'w') as f:
                f.write(f"{prediction}")
            output = prediction[-1]['generated_text'][-1]['content']
            keyword = '</think>'
            pre, sep, post =  output.partition(keyword)
            if sep: 
                coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", post)
                speed_curvature_pred = [[float(v), float(k)] for v, k in coordinates]
                if len(speed_curvature_pred) > 10: speed_curvature_pred = speed_curvature_pred[:10]
                print(f"Predictions for frame {i}/{scene_length - TOTAL_LENGTH}: {speed_curvature_pred}")

            pred_len = min(FUTURE_LENGHT, len(speed_curvature_pred))
            pred_curvatures = np.array(speed_curvature_pred)[:, 1] / 100
            pred_speeds = np.array(speed_curvature_pred)[:, 0]
            pred_traj = np.zeros((pred_len, 3))
            pred_traj[:pred_len, :2] = IntegrateCurvatureForPoints(pred_curvatures,pred_speeds,future_start_world,atan2(past_ego_velocities[-1][1],past_ego_velocities[-1][0]), pred_len)

            check_flag = OverlayTrajectory(img, pred_traj.tolist(), past_camera_params[-1], past_ego_poses[-1], color=(255, 0, 0), args=args)
            cam_images_sequence.append(img.copy())
            cv2.imwrite(f"car_results/deepseek/{timestamp}/{name}/img_{i}.jpg", img)

            # Compute ADE.
            future_ego_traj_world = np.array(future_ego_traj_world)
            ade = np.mean(np.linalg.norm(future_ego_traj_world[:pred_len] - pred_traj, axis=1))
            
            pred1_len = min(pred_len, 2)
            ade1s = np.mean(np.linalg.norm(future_ego_traj_world[:pred1_len] - pred_traj[1:pred1_len+1] , axis=1))
            ade1s_list.append(ade1s)

            pred2_len = min(pred_len, 4)
            ade2s = np.mean(np.linalg.norm(future_ego_traj_world[:pred2_len] - pred_traj[:pred2_len] , axis=1))
            ade2s_list.append(ade2s)

            pred3_len = min(pred_len, 6)
            ade3s = np.mean(np.linalg.norm(future_ego_traj_world[:pred3_len] - pred_traj[:pred3_len] , axis=1))
            ade3s_list.append(ade3s)
        
        WriteImageSequenceToVideo(cam_images_sequence, f"car_results/deepseek/{timestamp}/{name}/{name}")