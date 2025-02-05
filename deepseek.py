# Use a pipeline as a high-level helper
import os
import cv2
import argparse
import numpy as np
from nuscenes import NuScenes
from truckscenes import TruckScenes
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor, AutoProcessor, pipeline
from openemma.YOLO3D.inference import yolo3d_nuScenes



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek")
    parser.add_argument("--dataroot", type=str, default="/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes")
    parser.add_argument("--model", type=str, default="deepseek")