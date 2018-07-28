# -*- coding: utf-8 -*-

import glob

import matplotlib.pylab as plt
import numpy as np

import cv2
from dsb.conf import IMAGES_PATH
from dsb.utils import combine_masks, preprocess_image

# TODO: Processing pipeline for the images and masks.


def process_images():
    # TODO: Add some documentation
    images = []
    masks = []
    # Get the raw images
    for img_path in glob.iglob(IMAGES_PATH):
        print(img_path)
        img = preprocess_image(img_path)
        combined_mask = combine_masks(img_path)
        print(img)
        images.append(img)
        masks.append(combined_mask)
    return images, masks
