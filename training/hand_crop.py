"""Hand-focused classification views from supplied isolated-hand silhouettes.

This is label-preserving crop augmentation, not object-detection annotation.
Ambiguous/non-white-background inputs are left unchanged.
"""
import random

import numpy as np
from PIL import Image
from training.background_augmentation import white_background_mask


def hand_focused_crop(image: Image.Image, cutoff: float = .45) -> Image.Image:
    mask = white_background_mask(image)
    if mask is None:
        return image.copy()
    y, x = np.nonzero(~mask)
    points = np.column_stack((x, y))
    boundary = (x <= 1) | (x >= image.width-2) | (y <= 1) | (y >= image.height-2)
    if boundary.sum() < 3:
        return image.copy()
    wrist_root = points[boundary].mean(axis=0)
    direction = points.mean(axis=0) - wrist_root
    length = np.linalg.norm(direction)
    if length < 10:
        return image.copy()
    projection = (points-wrist_root) @ (direction/length)
    distal = points[projection >= cutoff*projection.max()]
    if len(distal) < 100:
        return image.copy()
    low, high = distal.min(axis=0), distal.max(axis=0)
    padding = .15*max(high-low)
    box = (max(0, int(low[0]-padding)), max(0, int(low[1]-padding)),
           min(image.width, int(high[0]+padding+1)), min(image.height, int(high[1]+padding+1)))
    return image.crop(box)


class HandCropVariation:
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= .65:
            return image
        return hand_focused_crop(image, random.uniform(.35, .5))
