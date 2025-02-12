import torch
from transformers import AutoModelForCausalLM

from DeepSeek_VL2.deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from DeepSeek_VL2.deepseek_vl2.utils.io import load_pil_images


# specify the path to the model
model_path = "deepseek-ai/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

## single image conversation example
## Please note that <|ref|> and <|/ref|> are designed specifically for the object localization feature. These special tokens are not required for normal conversations.
## If you would like to experience the grounded captioning functionality (responses that include both object localization and reasoning), you need to add the special token <|grounding|> at the beginning of the prompt. Examples could be found in Figure 9 of our paper.
conversation = [
    {
        "role": "<|User|>",
        "content": "<image>\n You are an autonomous driving labeller, give me all of the important information with regards to self driving in this scene.",
        "images": ["/home/ubuntu/project_ws/OpenEMMA/datasets/nuscenes/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402932162460.jpg"],
    },
    {"role": "<|Assistant|>", "content": ""},
]


# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True,
    system_prompt=""
).to(vl_gpt.device)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# run the model to get the response
outputs = vl_gpt.language.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
print(f"{prepare_inputs['sft_format'][0]}", answer)