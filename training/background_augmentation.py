"""Training-only background replacement for supplied isolated-hand images.

Masks are computed only from near-white pixels connected to the image boundary.
They are preprocessing masks, never fabricated object-detection annotations.
"""
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def white_background_mask(image: Image.Image) -> np.ndarray | None:
    pixels = np.asarray(image.convert('RGB'))
    white = (pixels.min(axis=2) >= 248) & (np.ptp(pixels, axis=2) <= 6)
    if white.mean() < .25:
        return None
    _, labels = cv2.connectedComponents(white.astype(np.uint8), connectivity=8)
    boundary = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    boundary = boundary[boundary != 0]
    background = np.isin(labels, boundary)
    if background.mean() < .25:
        return None
    return background


def patterned_background(size: tuple[int, int], rng: random.Random, style: str) -> Image.Image:
    color = tuple(rng.randint(25, 225) for _ in range(3))
    image = Image.new('RGB', size, color)
    if style == 'clutter':
        draw = ImageDraw.Draw(image)
        for _ in range(18):
            x, y = rng.randrange(size[0]), rng.randrange(size[1])
            w, h = rng.randint(10, max(10, size[0] // 3)), rng.randint(10, max(10, size[1] // 3))
            fill = tuple(rng.randint(20, 235) for _ in range(3))
            draw.rectangle((x, y, x + w, y + h), fill=fill)
        image = image.filter(ImageFilter.GaussianBlur(2))
    return image


def replace_background(image: Image.Image, background: Image.Image) -> Image.Image:
    image = image.convert('RGB')
    mask = white_background_mask(image)
    if mask is None:
        return image.copy()
    pixels = np.array(image)
    replacement = np.asarray(background.convert('RGB').resize(image.size))
    pixels[mask] = replacement[mask]
    return Image.fromarray(pixels)


class BackgroundVariation:
    """Augment existing training images; no new independent samples are implied."""
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= .7:
            return image
        # Only isolated views qualify. Real-background originals are retained.
        if white_background_mask(image) is None:
            return image
        background = patterned_background(image.size, random, random.choice(['solid', 'clutter']))
        return replace_background(image, background)


class ResolutionVariation:
    """Simulate reduced hand detail without cropping or changing finger positions."""
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= .4:
            return image
        factor = random.uniform(.3, .65)
        reduced = image.resize((max(1, int(image.width*factor)), max(1, int(image.height*factor))),
                               Image.Resampling.BILINEAR)
        return reduced.resize(image.size, Image.Resampling.BILINEAR)
