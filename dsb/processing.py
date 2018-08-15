# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
from tqdm import tqdm

from dsb.conf import IMAGES_BASE_PATH, TEST_IMAGE_IDS, TRAIN_IMAGE_IDS
from dsb.utils import combine_masks, preprocess_image

# TODO: Processing pipeline for the images and masks.

# TODO: Refactor the train and test processing pipelines.


def process_train_data(debug):
    # TODO: Add some documentation
    images = []
    masks = []
    # TODO: Improve this.
    if debug:
        # Useful while implementing the pipeline.
        ids = set(list(TRAIN_IMAGE_IDS)[0:10])
    else:
        ids = TRAIN_IMAGE_IDS
    # Get the raw images
    for img_id in tqdm(ids, desc="Image processing"):
        # Transform the pathlib path into a string so that cv2 works.
        img_path = str(Path(IMAGES_BASE_PATH) / img_id / 'images' / (img_id + '.png'))
        img = preprocess_image(img_path)
        img_name = Path(img_path).name
        masks_folder = Path(IMAGES_BASE_PATH) / img_id / 'masks'
        masks_paths = masks_folder.glob("*.png")
        combined_mask = combine_masks(masks_paths, img_name)
        images.append(img)
        masks.append(combined_mask)
    # Stack and add a new dimension at the first dimension so that the channels dim is the last one.
    # This is done so that it works wit TF convention (channels last).
    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)
    return images, masks


def process_test_data(debug):
    # TODO: Add some documentation
    images = []
    # Notice that I store the tests original image sizes for later upsampling during submission step.
    sizes = []
    # TODO: Improve this.
    if debug:
        # Useful while implementing the pipeline.
        ids = set(list(TEST_IMAGE_IDS)[0:10])
    else:
        ids = TEST_IMAGE_IDS
    # Get the raw images
    for img_id in tqdm(ids, desc="Image processing"):
        # Transform the pathlib path into a string so that cv2 works.
        img_path = str(Path(IMAGES_BASE_PATH) / img_id / 'images' / (img_id + '.png'))
        img = preprocess_image(img_path)
        images.append(img)
        # Append the original image shape
        sizes.append([img.shape[0], img.shape[1]])
    # Stack and add a new dimension at the first dimension so that the channels dim is the last one.
    # This is done so that it works wit TF convention (channels last).
    images = np.stack(images, axis=0)
    return images, sizes
