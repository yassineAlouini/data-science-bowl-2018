import glob
import os

import matplotlib.pylab as plt
import numpy as np
from tqdm import tqdm

import cv2
from dsb.conf import IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH

# TODO: Later, check how to do some of the processing steps directly with Keras using:
# https://keras.io/preprocessing/image/


def combine_masks(masks_paths, image_name):
    """ Combine the different masks of a single image into one mask
    """
    masks = []
    for mask_path in tqdm(masks_paths, desc='Masks for {} procesing'.format(image_name), leave=False):
        # Transform into string so that cv2 can read the file.
        # Only 1 channel for the masks.
        mask = preprocess_image(str(mask_path), channels=1)
        # mask = np.expand_dims(mask, axis=-1)
        masks.append(mask)
    if masks:
        return np.maximum.reduce(masks)
    return None


def preprocess_image(img_path, channels=IMG_CHANNELS):
    """ Preprocess an image given its path.
    """
    img = cv2.imread(img_path)
    img = img[:, :, :IMG_CHANNELS]
    img = np.resize(img, (IMG_HEIGHT, IMG_WIDTH, channels))
    return img


def plot_one_image(img_path):
    """Plot one image with its corresponding masks.
    """
    # TODO: Improve the paths building using pathlib.
    masks_folder = os.path.abspath(os.path.join(img_path, os.pardir)).replace('images', 'masks')
    masks_paths = glob.glob(os.path.join(masks_folder, '*.png'))
    img_name = os.path.basename(os.path.splitext(img_path)[0])
    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    img = cv2.imread(img_path)
    axes[0].imshow(img)
    axes[0].set_title('Image')
    mask = combine_masks(masks_paths)
    axes[1].imshow(mask)
    fig.suptitle(img_name)
    axes[1].set_title('Mask')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig
