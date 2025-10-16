# image_preprocessing.py
import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from PIL import Image
import numpy as np

# Constants from Qwen3-VL
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
SPATIAL_MERGE_SIZE = 2
PATCH_SIZE = 16
FACTOR = PATCH_SIZE * SPATIAL_MERGE_SIZE  # = 32
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def smart_resize(
    height: int,
    width: int,
    factor: int = FACTOR,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Resizes image while preserving aspect ratio and ensuring dimensions are divisible by `factor`.
    Mimics Qwen3-VL's official preprocessing.
    """
    max_pixels = max_pixels or (IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels or (IMAGE_MIN_TOKEN_NUM * factor ** 2)
    assert max_pixels >= min_pixels

    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"Aspect ratio too extreme (> {MAX_RATIO})")

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return h_bar, w_bar

def preprocess_image_for_qwen3vl(pil_image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocesses a PIL image for Qwen3VLVisionModel.
    
    Returns:
        pixel_values: [1, 3, 1, H, W] (normalized)
        grid_thw: [1, 3] = [1, H//16, W//16]
    """
    # Convert to RGB
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Get original size
    width, height = pil_image.size

    # Smart resize
    resized_height, resized_width = smart_resize(height, width, factor=FACTOR)

    # Resize image
    resized_image = pil_image.resize((resized_width, resized_height), Image.BICUBIC)

    # Convert to tensor [C, H, W] in [0, 1]
    img_tensor = torch.from_numpy(np.array(resized_image)).permute(2, 0, 1).float() / 255.0

    # Normalize with ImageNet stats (Qwen3-VL uses these)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
    img_tensor = (img_tensor - mean) / std

    # Add batch and temporal dims: [1, 3, 1, H, W]
    pixel_values = img_tensor.unsqueeze(0).unsqueeze(2)

    # Compute grid_thw: [1, H//16, W//16]
    grid_thw = torch.tensor([[1, resized_height // PATCH_SIZE, resized_width // PATCH_SIZE]], dtype=torch.long)

    return pixel_values, grid_thw