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
        max_new_tokens=512,
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
            "content": f"<image_placeholder>\n You are a autonomous driving labeller. You have access to these front-view camera images of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Describe the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings.",
            "images": img,
            },
            {"role": "Assistant", "content": ""},
        ]
    elif task == "object":
        prompt = [{
            "role": "User",
            "content": f"<image_placeholder>\n You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. What other road users should you pay attention to in the driving scene? List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you.",
            "images": img,
            },
            {"role": "Assistant", "content": ""},
        ] 
    elif task == "intent":
        if message == None:
            prompt = [{
                "role": "User",
                "content": f"<image_placeholder>\n You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, describe the desired intent of the ego car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?",
                "images": img,
                },
                {"role": "Assistant", "content": ""},
            ]
        else:
            prompt = [{
                "role": "User",
                "content": f"<image_placeholder>\n You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Half a second ago your intent was to {prev_intent}. Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? Explain your current intent: ",
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
        message (str): The prompt to pass to the LLM (DeepSeek-R1-Distill-Qwen-7B)
        llm_pipe (pipeline): The pipeline object that contains the LLM
    Returns:
        answer (str): The LLM's response to the prompt
    """
    prompt = [{"role": "user", "content": f"{message}"}]
    with open("prompt.txt", 'w') as f:
        f.write(f"{prompt}")
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
    if verbose: print(f"Scene Description done: \n {scene_description}")
    object_description = call_vlm(message=None, img=current_image, chat_processor=chat_processor, model=model, task="object")
    if verbose: print(f"Object Description done: \n{object_description}")
    intent_description = call_vlm(message=given_intent, img=current_image, chat_processor=chat_processor, model=model, task="intent")
    if verbose: print(f"Intent Description done: \n{intent_description}")
    
    past_waypoints_str = [f"[{x[0]:.2f},{x[1]:.2f}]" for x in past_waypoints]
    past_waypoints_str = ", ".join(past_waypoints_str)
    past_velocities_norm = np.linalg.norm(past_velocities, axis=1)
    past_curvatures = past_curvatures * 100
    past_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(past_velocities_norm, past_curvatures)]
    past_speed_curvature_str = ", ".join(past_speed_curvature_str)
    
    message = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
    The scene is described as follows: {scene_description}. 
    The identified critical objects are {object_description}. 
    The car's intent is {intent_description}. 
    The 5 second historical velocities and curvatures of the ego car are {past_speed_curvature_str}. 
    Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10] up to a maximum of 10. In the end provide a consice list of the output speeds and velocities in the same format as the historical velocities and curvatures in raw text, not markdown or latex. Future speeds and curvatures:"""
    if llm_pipe == None:
        print(f"Message that will be passed to VLM: \n{message}")
        final = call_vlm(message=message, img=current_image, chat_processor=chat_processor, model=model, task="final")
    else:
        print(f"Message that will be passed to LLM: \n{message}")
        final = call_llm(message=message, llm_pipe=llm_pipe)

    return final, scene_description, object_description, intent_description


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="janus1")
    parser.add_argument("--dataroot", type=str, default="/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes")
    parser.add_argument("--version", type=str, default='v1.0-mini')
    parser.add_argument("--vehicle", type=str, default='car')
    parser.add_argument("--verbose", type=bool, default=False)
    args = parser.parse_args()
    if args.model == "janus1":
        model_path = "deepseek-ai/Janus-Pro-1B"
    elif args.model== "janus7":
        model_path = "deepseek-ai/Janus-Pro-7B"

    # max_memory = {0:f"{(torch.cuda.get_device_properties('cuda:0').total_memory//(1024 ** 3)) * 0.8}GiB"}
    
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    llm_pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device=0, max_new_tokens=2048)
    if torch.cuda.is_available():
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda()
    
    # img_path = ["/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402932162460.jpg"], "/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes/sweeps/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402932612460.jpg"]
    
    # with open(os.path.join(img_path), "rb") as img_file:
    #     img = cv2.imdecode(np.frombuffer(img_file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    #     cv2.imwrite("testimg.jpg", img)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot)
    scenes = nusc.scene
    scene_list = ["scene-0061"]#["scene-0103", "scene-1077"]

    for scene in scenes:
        name = scene['name']
        if not name in scene_list:
            continue

        # Load all of the useful token info
        token = scene['token']
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        description = scene['description']

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
            
            prediction, scene_description, object_description, intent_description = GenerateMotion(current_image=[current_image], past_waypoints=past_ego_traj_world, past_velocities=past_ego_velocities, past_curvatures=past_ego_curvatures, given_intent=prev_intent, chat_processor=vl_chat_processor, model=vl_gpt, llm_pipe=llm_pipe, verbose=args.verbose)
            print(f"FINAL OUTPUT: \n{prediction}")
            print(f"FINAL OUTPUT[-1]: \n{prediction[-1]}")
            with open("test.txt", 'w') as f:
                f.write(f"Final Output: {prediction}")
            break

    # answer = describe_scene(img_path=img_path, chat_processor=vl_chat_processor, model=vl_gpt).strip('\n')

    # conversation = [{
    #     "role": "User",
    #     "content": "<image_placeholder>\nYou are a autonomous driving labeller. Imagine you are driving the car. You need to detect all of the objects that are in the image thatyou need to take into account to drive safely through the scene. List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you.",
    #     "images": [img_path],
    #     },
    #     {"role": "Assistant", "content": ""},
    # ]
    
    # answer, full_answer = vlm_inference(conversation, vl_chat_processor, vl_gpt)
    # print(f"Object Detection: \n{answer}")


    # print(answer)