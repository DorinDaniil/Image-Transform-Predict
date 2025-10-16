# extract_vision_weights.py
from transformers import Qwen3VLForConditionalGeneration
import torch

def extract_vision_weights():
    print("Loading full Qwen3-VL model...")
    full_model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct",
        device_map=None,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )

    vision_model = full_model.visual
    vision_state_dict = vision_model.state_dict()

    torch.save(vision_state_dict, "checkpoints/qwen3_vl_vision/qwen3_vl_vision_weights.pth")
    print("Vision weights saved to qwen3_vl_vision_weights.pth")

if __name__ == "__main__":
    extract_vision_weights()