# Inspired from this thread: https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741.

# TODO: Finish this.
import numpy as np


def label_mask(mask_img, border_img, seed_ths, threshold, seed_size=8, obj_size=10):
    img_copy = np.copy(mask_img)
    m = img_copy * (1 - border_img)
    img_copy[m <= seed_ths] = 0
    img_copy[m > seed_ths] = 1
    img_copy = img_copy.astype(np.bool)
    img_copy = remove_small_objects(img_copy, seed_size).astype(np.uint8)
    mask_img[mask_img <= threshold] = 0
    mask_img[mask_img > threshold] = 1
    mask_img = mask_img.astype(np.bool)
    mask_img = remove_small_objects(mask_img, obj_size).astype(np.uint8)
    markers = ndimage.label(img_copy, output=np.uint32)[0]
    labels = watershed(mask_img, markers, mask=mask_img, watershed_line=True)
    return labels


# Run length encoding inspired from here: https://www.kaggle.com/rakhlin/fast-run-length-encoding-python

# TODO: Improve this

def rle_encoding(x):
    """
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    """
    dots = np.where(x.T.flatten() == 1)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
