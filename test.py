import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
tokenizer=None


image = '/home/ubuntu/project_ws/OpenEMMA/lane_test/frame_00002540.jpg'
prompt = "I want you to analyze this image in detail, then pick one of the lane dividing lines that you see in this image. From this lane dividing line, produce a polyline object that perfectly represents that dividing line. The polyline should be based off pixel locations in the image"
message = [{"role": "user",
            "content": [{"type": "image", "image": image},{"type": "text", "text": prompt}]}]
text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(message)
inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",).to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=2048)
generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output_text[0])