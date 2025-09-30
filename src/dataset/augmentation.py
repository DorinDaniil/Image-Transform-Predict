import random
from typing import List, Tuple
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from torchvision.transforms import functional

class ImageTransformer:
    """
    A class to apply random sequences of image transformations.
    Each transformation is sampled with a given probability, and rotations are mutually exclusive.
    """
    transform_tokens = {
        "noop": 3,            # Identity transformation (no change)
        "grayscale": 4,       # Convert to grayscale
        "rotate_90": 5,       # Rotate 90 degrees clockwise
        "rotate_180": 6,      # Rotate 180 degrees
        "rotate_270": 7,      # Rotate 270 degrees clockwise
        "color_jitter": 8,    # Randomly adjust color
        "noise_adding": 9,    # Add random noise
        "crop": 10,           # Random crop (non-central)
        "horizontal_flip": 11, # Flip horizontally
        "vertical_flip": 12,  # Flip vertically
        "resize": 13,         # Random resize
    }

    def __init__(self):
        """
        Initialize the transformer.
        """
        self.transformations = list(self.transform_tokens.keys())

    def resize(self, image: Image.Image) -> Image.Image:
        """
        Resize the image randomly, either uniformly or with different scales for width and height.
        Args:
            image (Image.Image): Input image.
        Returns:
            Image.Image: Resized image.
        """
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
        """
        Randomly crop the image, with the crop size as a percentage of the original size.
        The crop is not necessarily central.
        Args:
            image (Image.Image): Input image.
            min_percent (float): Minimum crop percent of the original size (default: 1).
            max_percent (float): Maximum crop percent of the original size (default: 7).
        Returns:
            Image.Image: Cropped image.
        """
        size = random.uniform(min_percent, max_percent)
        w, h = image.size
        left = random.randint(0, int(w * size / 100))
        top = random.randint(0, int(h * size / 100))
        width = random.randint(int(w * (1 - size / 100) - left), int(w - left - 1))
        height = random.randint(int(h * (1 - size / 100) - top), int(h - top - 1))
        return functional.crop(image, top=top, left=left, height=height, width=width)

    def apply_transform(self, image: Image.Image, transform: str) -> Image.Image:
        """
        Apply a single transformation to the image.
        Args:
            image (Image.Image): Input image.
            transform (str): Name of the transformation to apply.
        Returns:
            Image.Image: Transformed image.
        """
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

    def sample_transformations(self, p: float = 0.4) -> List[str]:
        """
        Sample a random sequence of transformations, ensuring rotations are mutually exclusive.
        Args:
            p (float): Probability of selecting each transformation (default: 0.4).
        Returns:
            List[str]: List of selected transformations.
        """
        selected = []
        rotations = ["rotate_90", "rotate_180", "rotate_270"]
        rotation_chosen = False
        for transform in self.transformations:
            if transform == "noop":
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
        """
        Apply a random sequence of transformations to the image.
        Args:
            image (Image.Image): Input image.
        Returns:
            Tuple[Image.Image, List[str]]: Transformed image and the sequence of applied transformations.
        """
        sequence = self.sample_transformations()
        transformed_image = image.copy()
        for transform in sequence:
            transformed_image = self.apply_transform(transformed_image, transform)
        return transformed_image, sequence