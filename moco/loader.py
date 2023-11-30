# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random

import numpy as np
from PIL import ImageFilter, ImageOps
from albumentations.core.composition import BaseCompose


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, second_transform):
        self.q_transform = base_transform
        self.k_transform = base_transform
        if second_transform is not None:
            self.k_transform = second_transform

    def __call__(self, **kwargs):
        q = self.q_transform(**kwargs)
        k = self.k_transform(**kwargs)
        q_mask, k_mask = None, None
        if isinstance(self.q_transform, BaseCompose):
            if 'masks' in q:
                q_mask, q_segmentation = q['masks']
                k_mask, k_segmentation = k['masks']
            elif 'mask' in q:
                q_mask, q_segmentation = q['mask'], None
                k_mask, k_segmentation = k['mask'], None
            else:
                q_mask, q_segmentation = None, None
                k_mask, k_segmentation = None, None
            if q_mask is not None:
                q_mask = q_mask.astype(bool)
            if q_segmentation is not None:
                q_segmentation = q_segmentation.astype(bool)
            if k_mask is not None:
                k_mask = k_mask.astype(bool)
            if k_segmentation is not None:
                k_segmentation = k_segmentation.astype(bool)
            q, k = q['image'], k['image']
        return tuple(([q, k], [q_mask, k_mask], [q_segmentation, k_segmentation]))


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization(object):
    """Solarize image"""

    def __init__(self, magnitude):
        self.magnitude = magnitude

    def __call__(self, x, magnitude):
        magnitudes = np.linspace(0, 256, 11)
        # selected_magnitudes = np.random.choice(magnitudes, size=2, replace=False)
        img = ImageOps.solarize(x, random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
        return img
