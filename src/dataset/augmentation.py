import random
import numpy as np
from typing import List, Tuple
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from torchvision.transforms import functional

class ImageTransformer:
    transform_tokens = {
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
        "resize": 13,
    }

    def __init__(self):
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

    def crop(self, image: Image.Image, min_percent: int = 3, max_percent: int = 10):
        size = random.uniform(min_percent, max_percent)
        w, h = image.size
        left = random.randint(0, int(w * size / 100))
        top = random.randint(0, int(h * size / 100))
        width = random.randint(int(w * (1 - size / 100) - left), int(w - left - 1))
        height = random.randint(int(h * (1 - size / 100) - top), int(h - top - 1))
        return functional.crop(image, top=top, left=left, height=height, width=width)

    import numpy as np

    def is_grayscale(self, image: Image.Image) -> bool:
        if image.mode in ("L", "LA", "P"):
            return True

        img_array = np.array(image)

        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            return np.allclose(img_array[:, :, 0], img_array[:, :, 1]) and \
                np.allclose(img_array[:, :, 1], img_array[:, :, 2])

        return False

    def apply_transform(self, image: Image.Image, transform: str) -> Image.Image:
        if transform == "noop":
            return image
        elif transform == "grayscale":
            if self.is_grayscale(image):
                return image  # Do not use grayscale if the image is already black and white
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
            return image.point(lambda p: p * random.uniform(0.9, 1.1))
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

    def sample_transformations(self, image: Image.Image, p: float = 0.4) -> List[str]:
        selected = []
        rotations = ["rotate_90", "rotate_180", "rotate_270"]
        rotation_chosen = False

        is_gray = self.is_grayscale(image)

        for transform in self.transformations:
            if transform == "noop":
                continue
            if transform == "grayscale" and is_gray:
                continue
            if transform in rotations:
                if not rotation_chosen and random.random() < p:
                    selected.append(transform)
                    rotation_chosen = True
            else:
                if random.random() < p:
                    selected.append(transform)

        return selected if selected else ["noop"]

    def transform(self, image: Image.Image) -> Tuple[Image.Image, List[str]]:
        sequence = self.sample_transformations(image)
        transformed_image = image.copy()
        for transform in sequence:
            transformed_image = self.apply_transform(transformed_image, transform)
        return transformed_image, sequence