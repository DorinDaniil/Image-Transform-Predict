"""
Padding / framing: the inverse of crop.

In v2 the ``crop`` token (id 10, same as v1) means "the field of view changed":
either part of the image was cut away (v1 crop) or margins were ADDED around it
(a frame, document margins, letterboxing). Both are produced by the same token
because, as evidence, they say the same thing -- the copy shows a different
extent of the same picture -- and because adding a frame and cropping are each
other's inverse, so two separate tokens would be systematically confusable.

Padding styles (``PAD_STYLES`` with weights):
    solid        uniform colour margins: white (document / slide), black, grey,
                 the image's own mean colour, or a saturated colour
    blur_extend  margins filled with a heavily blurred, enlarged copy of the
                 image (the social-media "blurred background" look)
    mirror       margins filled by reflecting the image edges
Geometry: uniform margins on all sides (most common) or letterbox-style margins
on one axis only (fitting a 16:9 / 4:3 / square slot).
"""

import random
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

PAD_STYLES: Tuple[str, ...] = ("solid", "blur_extend", "mirror")
DEFAULT_PAD_STYLE_WEIGHTS: Dict[str, float] = {"solid": 6.0, "blur_extend": 2.0, "mirror": 1.0}

_PALETTE = [(230, 40, 40), (255, 140, 0), (255, 215, 0), (40, 170, 70), (30, 120, 220), (140, 60, 200)]


def _margins(w: int, h: int) -> Tuple[int, int, int, int]:
    """(left, top, right, bottom) margins in pixels."""
    r = random.random()
    if r < 0.6:  # uniform frame
        f = random.uniform(0.02, 0.12)
        m = max(2, int(min(w, h) * f))
        return m, m, m, m
    if r < 0.85:  # letterbox: one axis only, like fitting a slot of another aspect ratio
        f = random.uniform(0.06, 0.28)
        if random.random() < 0.5:
            m = max(2, int(h * f))
            top = random.randint(0, m) if random.random() < 0.3 else m // 2
            return 0, top, 0, m - top
        m = max(2, int(w * f))
        left = random.randint(0, m) if random.random() < 0.3 else m // 2
        return left, 0, m - left, 0
    # irregular margins (sloppy screenshot / scan)
    return tuple(max(0, int(min(w, h) * random.uniform(0.0, 0.12))) for _ in range(4))  # type: ignore[return-value]


def _solid_colour(img: Image.Image) -> Tuple[int, int, int]:
    r = random.random()
    if r < 0.5:
        return (255, 255, 255)
    if r < 0.65:
        return (0, 0, 0)
    if r < 0.75:
        g = random.randint(90, 230)
        return (g, g, g)
    if r < 0.9:
        mean = np.asarray(img.resize((16, 16), Image.BILINEAR), np.float32).reshape(-1, 3).mean(0)
        return tuple(int(v) for v in mean)
    return random.choice(_PALETTE)


def apply_pad(image: Image.Image, style_weights: Optional[Dict[str, float]] = None) -> Image.Image:
    img = image.convert("RGB")
    w, h = img.size
    left, top, right, bottom = _margins(w, h)
    if left + right + top + bottom < 4:
        left = top = right = bottom = max(2, int(min(w, h) * 0.04))
    W, H = w + left + right, h + top + bottom

    weights = dict(DEFAULT_PAD_STYLE_WEIGHTS)
    if style_weights:
        weights.update({str(k): float(v) for k, v in style_weights.items()})
    names = [k for k, v in weights.items() if v > 0] or ["solid"]
    style = random.choices(names, weights=[weights[k] for k in names])[0]

    if style == "blur_extend":
        bg = img.resize((W, H), Image.BILINEAR).filter(ImageFilter.GaussianBlur(max(4.0, min(W, H) * 0.04)))
        if random.random() < 0.5:  # darken a bit, as apps do
            bg = Image.fromarray((np.asarray(bg, np.float32) * random.uniform(0.6, 0.9)).astype(np.uint8), "RGB")
        canvas = bg
    elif style == "mirror":
        arr = np.asarray(img)
        # numpy reflect cannot pad more than the image size; clip margins to it
        t, b = min(top, h - 1), min(bottom, h - 1)
        l, r = min(left, w - 1), min(right, w - 1)
        arr = np.pad(arr, ((t, b), (l, r), (0, 0)), mode="reflect")
        canvas = Image.fromarray(arr, "RGB")
        if canvas.size != (W, H):  # finish with a solid border if margins were clipped
            full = Image.new("RGB", (W, H), _solid_colour(img))
            full.paste(canvas, ((W - canvas.width) // 2, (H - canvas.height) // 2))
            canvas = full
        return canvas
    else:
        canvas = Image.new("RGB", (W, H), _solid_colour(img))

    canvas.paste(img, (left, top))
    if style == "solid" and random.random() < 0.25:  # thin line around the picture
        from PIL import ImageDraw
        lw = max(1, int(min(w, h) * 0.003))
        col = (0, 0, 0) if sum(canvas.getpixel((0, 0))) > 384 else (255, 255, 255)
        ImageDraw.Draw(canvas).rectangle([left - lw, top - lw, left + w + lw - 1, top + h + lw - 1], outline=col, width=lw)
    return canvas
