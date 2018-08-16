import glob
import os

import matplotlib.pylab as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

from dsb.conf import IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH

# TODO: Later, check how to do some of the processing steps directly with Keras using:
# https://keras.io/preprocessing/image/


def combine_masks(masks_paths, image_name):
    """ Combine the different masks of a single image into one mask
    """
    masks = []
    for mask_path in tqdm(masks_paths, desc='Processing masks for image {}'.format(image_name), leave=False):
        # Transform into string so that skimage.io can read the file.
        mask = preprocess_image(str(mask_path), is_mask=True)
        # Add the missing third axis for the mask (the channel dim)
        mask = np.expand_dims(mask, axis=-1)
        # Cast to integer (0 or 1 values for the mask)
        mask = mask.astype(int)
        masks.append(mask)
    return np.maximum.reduce(masks)


def preprocess_image(img_path, is_mask=False):
    """ Preprocess an image given its path.
    """
    img = imread(img_path)
    if not is_mask:
        img = img[:, :, :IMG_CHANNELS]
    # TODO: What about aliasing effects when downsampling?
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True, anti_aliasing=True)
    # Scale to the range [0, 1]
    img /= 255.0
    return img

# TODO: Update this function (make it work again).


def plot_one_image(img_path):
    """Plot one image with its corresponding masks.
    """
    # TODO: Improve the paths building using pathlib.
    masks_folder = os.path.abspath(os.path.join(img_path, os.pardir)).replace('images', 'masks')
    masks_paths = glob.glob(os.path.join(masks_folder, '*.png'))
    img_name = os.path.basename(os.path.splitext(img_path)[0])
    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    img = imread(img_path)
    axes[0].imshow(img)
    axes[0].set_title('Image')
    mask = combine_masks(masks_paths)
    axes[1].imshow(mask)
    fig.suptitle(img_name)
    axes[1].set_title('Mask')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig
