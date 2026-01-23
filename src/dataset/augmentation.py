import random
import numpy as np
import io
import torchvision.transforms as transforms
from typing import List, Tuple, Optional
from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms import functional


class AugmentationScheduler:
    """
    Controls augmentation probability p during training.
    Interface and behavior match torch.optim.lr_scheduler.MultiStepLR.

    Args:
        initial_p (float): Initial probability (used before any milestone).
        milestones (List[int]): Epoch indices (0-based) after which p changes.
        probs (List[float]): New probabilities corresponding to each milestone.

    Example:
        # p = 0.1 for epochs 0-4 (first 5 epochs)
        # p = 0.3 starting from epoch 5 (6th epoch in logs)
        scheduler = AugmentationScheduler(
            initial_p=0.1,
            milestones=[5, 10],
            probs=[0.3, 0.5]
        )
    """

    def __init__(
        self,
        initial_p: float = 0.3,
        milestones: Optional[List[int]] = None,
        probs: Optional[List[float]] = None,
    ):
        self.initial_p = initial_p
        self.milestones = milestones if milestones is not None else []
        self.probs = probs if probs is not None else []

        if len(self.milestones) != len(self.probs):
            raise ValueError("milestones and probs must have the same length")
        if not all(self.milestones[i] < self.milestones[i + 1] for i in range(len(self.milestones) - 1)):
            raise ValueError("milestones must be increasing")

        self.last_epoch = -1
        self._current_p = initial_p

    def step(self):
        """Call once per epoch (no arguments)."""
        self.last_epoch += 1
        p = self.initial_p
        for milestone, prob in zip(self.milestones, self.probs):
            if self.last_epoch >= milestone:
                p = prob
            else:
                break
        self._current_p = p

    @property
    def p(self) -> float:
        return self._current_p

    def state_dict(self) -> dict:
        return {
            "initial_p": self.initial_p,
            "milestones": self.milestones,
            "probs": self.probs,
            "last_epoch": self.last_epoch,
            "_current_p": self._current_p,
        }

    def load_state_dict(self, state: dict):
        self.initial_p = state["initial_p"]
        self.milestones = state["milestones"]
        self.probs = state["probs"]
        self.last_epoch = state["last_epoch"]
        self._current_p = state["_current_p"]


class ImageTransformer:
    """
    Applies a sequence of image transformations and returns the transformed image
    along with the list of applied transformation names.
    """

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
            "jpeg_artefacts": 13,
        }
        self.transformations = list(self.transform_tokens.keys())
        self._current_p = 0.3  # default fallback

    def set_p(self, p: float):
        """Set the current augmentation probability."""
        self._current_p = p

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
        quality = random.randint(20, 70)
        image.save(buffer, format='JPEG', quality=quality, optimize=True)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')

    def blur(self, image: Image.Image) -> Image.Image:
        radius = random.uniform(1.0, 3.0)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def compression(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        max_allowed = 223
        if w <= max_allowed and h <= max_allowed:
            scale = random.uniform(0.5, 0.95)
            new_w = max(32, int(w * scale))
            new_h = max(32, int(h * scale))
        else:
            scale = min(max_allowed / w, max_allowed / h)
            scale *= random.uniform(0.7, 1.0)
            new_w = max(32, int(w * scale))
            new_h = max(32, int(h * scale))
        return image.resize((new_w, new_h), Image.LANCZOS)

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
        elif transform == "compression":
            return self.compression(image)
        elif transform == "blur":
            return self.blur(image)
        else:
            return image

    def sample_transformations(self, image: Image.Image, p: float = 0.3) -> List[str]:
        """Randomly sample a subset of augmentations to apply to the input image.

        This method samples a combination of transformations while avoiding redundant or
        self-canceling sequences (i.e., combinations equivalent to a noop). The goal is to
        maintain a unique, non-degenerate mapping between the applied sequence and the
        resulting image, while allowing the model to learn meaningful relationships and
        compositional structure between transformations.

        At most one rotation (90, 180, or 270 degrees) is selected per sample to avoid
        overlapping orientation states. Both horizontal and vertical flips can be included
        in the same sequence, as their combination is not excluded unless it directly leads
        to a noop-equivalent transformation. Heavy distortions (e.g., JPEG artifacts, noise)
        are limited to one per sequence. Grayscale conversion is skipped if the image is
        already grayscale, and color jitter is omitted when grayscale is selected.

        Args:
            image: Input PIL image used to determine grayscale status.
            p: Probability threshold for including each transformation (default: 0.3).

        Returns:
            List of selected transformation names, or ["noop"] if none are chosen.
        """
        is_gray = self.is_grayscale(image)
        transforms_to_sample = [t for t in self.transformations if t != "noop"]
        random.shuffle(transforms_to_sample)

        selected = []
        flags = {
            "r180": False,
            "h": False,
            "v": False,
            "rot": False,
            "grayscale_selected": False,
            "heavy_distortion": False
        }

        heavy_distortions = {"jpeg_artefacts", "noise_adding"}

        for t in transforms_to_sample:
            if t == "grayscale" and is_gray:
                continue
            if random.random() >= p:
                continue
            
            if t == "grayscale":
                flags["grayscale_selected"] = True

            if t == "color_jitter":
                if flags["grayscale_selected"]:
                    continue
                selected.append(t)
                continue

            if t in heavy_distortions:
                if not flags["heavy_distortion"]:
                    selected.append(t)
                    flags["heavy_distortion"] = True
                continue

            if t in ("rotate_90", "rotate_270"):
                if not flags["rot"]:
                    selected.append(t)
                    flags["rot"] = True

            elif t == "rotate_180":
                if not flags["rot"] and not (flags["h"] and flags["v"]):
                    selected.append(t)
                    flags["r180"] = True
                    flags["rot"] = True

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

    def transform(self, image: Image.Image, p: Optional[float] = None) -> Tuple[Image.Image, List[str]]:
        actual_p = p if p is not None else self._current_p
        sequence = self.sample_transformations(image, actual_p)
        transformed_image = image.copy()
        for transform in sequence:
            transformed_image = self.apply_transform(transformed_image, transform)
        return transformed_image.convert('RGB'), sequence


    def sample_transformations_by_length(self, image: Image.Image, k: int) -> List[str]:
        """
        Sample exactly k transformations (1 <= k <= 5), respecting semantic constraints.
        
        Args:
            image: Input image (used to check if grayscale).
            k: Desired number of transformations (1 <= k <= 5).
            
        Returns:
            List of exactly k transformation names.
            If k == 1, may return ["noop"] with 5% probability.
            For k >= 2, NEVER includes "noop".
        """
        if not (1 <= k <= 5):
            raise ValueError("k must be between 1 and 5")
    
        is_gray = self.is_grayscale(image)
        transforms_to_sample = [t for t in self.transformations if t != "noop"]
        random.shuffle(transforms_to_sample)
    
        # Special case: k == 1
        if k == 1:
            if random.random() < 0.05:  # 5% chance of noop
                return ["noop"]
            for t in transforms_to_sample:
                if t == "grayscale" and is_gray:
                    continue
                # Check basic validity
                if t == "grayscale":
                    return [t]
                elif t == "color_jitter":
                    if not is_gray:
                        return [t]
                elif t in ("jpeg_artefacts", "noise_adding"):
                    return [t]
                elif t in ("rotate_90", "rotate_180", "rotate_270", "horizontal_flip", "vertical_flip", "crop"):
                    return [t]
            return ["noop"]
    
        # Case k >= 2
        selected = []
        flags = {
            "r180": False,
            "h": False,
            "v": False,
            "rot": False,
            "grayscale_selected": is_gray,
            "heavy_distortion": False
        }
    
        heavy_distortions = {"jpeg_artefacts", "noise_adding"}
    
        for t in transforms_to_sample:
            if len(selected) >= k:
                break
            if t in selected:  # ensure uniqueness
                continue
            if t == "grayscale" and is_gray:
                continue
    
            if t == "grayscale":
                selected.append(t)
                flags["grayscale_selected"] = True
    
            elif t == "color_jitter":
                if not flags["grayscale_selected"]:
                    selected.append(t)
    
            elif t in heavy_distortions:
                if not flags["heavy_distortion"]:
                    selected.append(t)
                    flags["heavy_distortion"] = True
    
            elif t in ("rotate_90", "rotate_270"):
                if not flags["rot"]:
                    selected.append(t)
                    flags["rot"] = True
    
            elif t == "rotate_180":
                if not flags["rot"] and not (flags["h"] and flags["v"]):
                    selected.append(t)
                    flags["r180"] = True
                    flags["rot"] = True
    
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
    
        # If couldn't reach k, try to fill with safe transforms (respecting constraints)
        while len(selected) < k:
            possible = []
            for t in transforms_to_sample:
                if t in selected:
                    continue
                if t == "grayscale" and is_gray:
                    continue
    
                valid = False
                if t == "grayscale":
                    valid = not flags["grayscale_selected"]
                elif t == "color_jitter":
                    valid = not flags["grayscale_selected"]
                elif t in heavy_distortions:
                    valid = not flags["heavy_distortion"]
                elif t in ("rotate_90", "rotate_270"):
                    valid = not flags["rot"]
                elif t == "rotate_180":
                    valid = not flags["rot"] and not (flags["h"] and flags["v"])
                elif t == "horizontal_flip":
                    valid = not (flags["r180"] and flags["v"])
                elif t == "vertical_flip":
                    valid = not (flags["r180"] and flags["h"])
                else:
                    valid = True
    
                if valid:
                    possible.append(t)
    
            if not possible:
                break
            t = random.choice(possible)
            selected.append(t)
            # Update flags
            if t == "grayscale":
                flags["grayscale_selected"] = True
            elif t in heavy_distortions:
                flags["heavy_distortion"] = True
            elif t in ("rotate_90", "rotate_270"):
                flags["rot"] = True
            elif t == "rotate_180":
                flags["r180"] = True
                flags["rot"] = True
            elif t == "horizontal_flip":
                flags["h"] = True
            elif t == "vertical_flip":
                flags["v"] = True
    
        # Final padding (fallback)
        while len(selected) < k:
            selected.append(selected[-1] if selected else "color_jitter")
    
        # Ensure no "noop" for k >= 2
        selected = [t for t in selected if t != "noop"]
        while len(selected) < k:
            selected.append("color_jitter")
    
        return selected[:k]

    def transform_by_length(self, image: Image.Image, k: int) -> Tuple[Image.Image, List[str]]:
        """Apply exactly k transformations to the image."""
        sequence = self.sample_transformations_by_length(image, k)
        transformed_image = image.copy()
        for transform in sequence:
            transformed_image = self.apply_transform(transformed_image, transform)
        return transformed_image.convert('RGB'), sequence