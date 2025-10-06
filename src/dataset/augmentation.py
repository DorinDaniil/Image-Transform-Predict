import random
import numpy as np
import io
import torchvision.transforms as transforms
from typing import List, Tuple
from PIL import Image, ImageOps
from torchvision.transforms import functional

class ImageTransformer:

    def __init__(self):
        self.transform_tokens = {
            "noop": 3,
            "grayscale": 4,
            "rotate_90": 5,
            "rotate_180": 6,
            "rotate_270": 7,
            "color_jitter": 8,
            "noise_adding": 9,
            "crop": 10,
            "horizontal_flip": 11,
            "vertical_flip": 12,
            # "resize": 13,
            # "jpeg_artefacts": 14,
        }
        self.transformations = list(self.transform_tokens.keys())

    def resize(self, image: Image.Image) -> Image.Image:
        if random.random() > 0.5:
            scale = random.uniform(0.3, 2)
            w, h = image.size
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            scale_w = random.uniform(0.3, 2)
            scale_h = random.uniform(0.3, 2)
            w, h = image.size
            new_w, new_h = int(w * scale_w), int(h * scale_h)
        return image.resize((new_w, new_h))

    def crop(self, image: Image.Image, min_percent: int = 3, max_percent: int = 15):
        size = random.uniform(min_percent, max_percent)
        w, h = image.size
        left = random.randint(0, int(w * size / 100))
        top = random.randint(0, int(h * size / 100))
        width = random.randint(int(w * (1 - size / 100) - left), int(w - left - 1))
        height = random.randint(int(h * (1 - size / 100) - top), int(h - top - 1))
        return functional.crop(image, top=top, left=left, height=height, width=width)

    def is_grayscale(self, image: Image.Image) -> bool:
        if image.mode in ("L", "LA", "P"):
            return True
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            return np.allclose(img_array[:, :, 0], img_array[:, :, 1]) and \
                np.allclose(img_array[:, :, 1], img_array[:, :, 2])
        return False

    def noise_adding(self, image: Image.Image) -> Image.Image:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_np = np.array(image).astype(np.float32)
        mean = 0
        sigma = random.uniform(5, 20)
        gaussian = np.random.normal(mean, sigma, img_np.shape)
        noisy = img_np + gaussian
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy, 'RGB')

    def jpeg_artefacts(self, image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        quality = random.randint(30, 70)
        image.save(buffer, format='JPEG', quality=quality, optimize=True)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')

    def apply_transform(self, image: Image.Image, transform: str) -> Image.Image:
        if transform == "noop":
            return image
        elif transform == "grayscale":
            return ImageOps.grayscale(image)
        elif transform == "rotate_90":
            return image.rotate(90, expand=True)
        elif transform == "rotate_180":
            return image.rotate(180, expand=True)
        elif transform == "rotate_270":
            return image.rotate(270, expand=True)
        elif transform == "color_jitter":
            enhancer = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
            return enhancer(image)
        elif transform == "noise_adding":
            return self.noise_adding(image)
        elif transform == "jpeg_artefacts":
            return self.jpeg_artefacts(image)
        elif transform == "crop":
            return self.crop(image)
        elif transform == "horizontal_flip":
            return ImageOps.mirror(image)
        elif transform == "vertical_flip":
            return ImageOps.flip(image)
        elif transform == "resize":
            return self.resize(image)
        else:
            return image

    def sample_transformations(self, image: Image.Image, p: float = 0.3) -> List[str]:
        is_gray = self.is_grayscale(image)
        transforms_to_sample = [t for t in self.transformations if t != "noop"]
        random.shuffle(transforms_to_sample)

        selected = []
        flags = {"r180": False, "h": False, "v": False}
        rot_selected = False

        for t in transforms_to_sample:
            if t == "grayscale" and is_gray:
                continue
            if random.random() >= p:
                continue

            if t in ("rotate_90", "rotate_270"):
                if not rot_selected:
                    selected.append(t)
                    rot_selected = True
            elif t == "rotate_180":
                if not rot_selected and not (flags["h"] and flags["v"]):
                    selected.append(t)
                    flags["r180"] = True
                    rot_selected = True
            elif t == "horizontal_flip":
                if not (flags["r180"] and flags["v"]):
                    selected.append(t)
                    flags["h"] = True
            elif t == "vertical_flip":
                if not (flags["r180"] and flags["h"]):
                    selected.append(t)
                    flags["v"] = True
            else:
                selected.append(t)

        return selected if selected else ["noop"]

    def transform(self, image: Image.Image, p: float = 0.3) -> Tuple[Image.Image, List[str]]:
        sequence = self.sample_transformations(image, p)
        transformed_image = image.copy()
        for transform in sequence:
            transformed_image = self.apply_transform(transformed_image, transform)
        return transformed_image.convert('RGB'), sequence