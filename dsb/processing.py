# -*- coding: utf-8 -*-

import glob
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dsb.conf import IMAGES_BASE_PATH, TRAIN_IMAGE_IDS
from dsb.utils import combine_masks, preprocess_image

# TODO: Processing pipeline for the images and masks.


def process_train_data():
    # TODO: Add some documentation
    images = []
    masks = []
    # Get the raw images
    for img_id in tqdm(TRAIN_IMAGE_IDS, desc="Image processing"):
        # Transform the pathlib path into a string so that cv2 works.
        img_path = str(Path(IMAGES_BASE_PATH) / img_id / 'images' / (img_id + '.png'))
        img = preprocess_image(img_path)
        img_name = Path(img_path).name
        masks_folder = Path(IMAGES_BASE_PATH) / img_id / 'masks'
        masks_paths = masks_folder.glob("*.png")
        combined_mask = combine_masks(masks_paths, img_name)
        images.append(img)
        masks.append(combined_mask)
    images = np.vstack(images)
    print(images.shape)
    print([m.shape for m in masks])
    masks = np.vstack(masks)
    print(masks.shape)
    return images, masks
