import cv2
import numpy as np
import glob
import matplotlib.pylab as plt


def combine_masks(masks_paths):
    """ Combine the different masks of a single image into one mask
    """
    masks = []
    for mask_path in masks_paths:
        mask = cv2.imread(mask_path)
        masks.append(mask)
    return np.maximum.reduce(masks)


def plot_one_image(img_path):
    """Plot one image with its corrsponding masks.
    """
    masks_folder = os.path.abspath(os.path.join(img_path, os.pardir)).replace('images', 'masks')
    masks_paths = glob.glob(os.path.join(masks_folder, '*.png'))
    img_name = ntpath.basename(os.path.splitext(img_path)[0])
    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    img = cv2.imread(img_path)
    axes[0].imshow(img)
    axes[0].set_title('Image')
    mask = combine_masks(masks_paths)
    axes[1].imshow(mask)
    fig.suptitle(img_name)
    axes[1].set_title('Mask')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
