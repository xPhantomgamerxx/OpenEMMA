# Use a pipeline as a high-level helper
import os
import cv2
import numpy as np
from nuscenes import NuScenes
from truckscenes import TruckScenes
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor, AutoProcessor, pipeline
from openemma.YOLO3D.inference import yolo3d_nuScenes

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
