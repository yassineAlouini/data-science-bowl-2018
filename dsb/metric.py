# Competition's metric implementation
# Check this Wikipedia page for more details about Jacard index:
# https://en.wikipedia.org/wiki/Jaccard_index

import glob

import numpy as np

import cv2
from dsb.conf import TEST_MASK_PATH

START = 0.5
END = 0.95
STEP = 0.05
N_STEPS = int((END - START) / STEP) + 2
THRESHOLDS = np.linspace(START, END, N_STEPS)


def _custom_precision(labels, total_true, total_pred, positive_class=1):
    """ A custom precision (i.e. different to the common one) metric for a two
    classes (positive and negative) classification.
    Notice that this custom precision is exactly the Jaccard similarity coefficient.
    """
    true_positive = (labels == positive_class).sum()
    print(true_positive)
    false_positive = total_pred - true_positive
    false_negative = total_true - true_positive
    return true_positive / (true_positive + false_positive + false_negative)


def _iou(img_1, img_2):
    """ IoU (intersection over union also called Jacard index) computation for two
    images. Notice that the pixels are transfored into binary values: either bigger than 0 (image) or not
    (background). The 0 value is the code for black color.
    """
    intersection = ((img_1 & img_2) > 0).sum()
    union = ((img_2 | img_2) > 0).sum()
    return intersection / union


def _labeling(pred_mask, true_mask, thresholds):
    """ Label the predicted mask in comparaison
    of the true one using the IoU (intersection over union) metric and the thresolds:
    1 if IoU > threshold, 0 otherwise.
    """
    computed_iou = _iou(pred_mask, true_mask)
    return (computed_iou > thresholds).astype(int)


def dsb_metric(pred_masks, true_masks, thresolds=THRESHOLDS):
    """ A specific IoU (intersection over union) for the dsb competition
    """
    total_true_masks = len(pred_masks)
    total_pred_masks = len(true_masks)
    mean_jacard_sim_coeff = 0.0
    # TODO: Something is off, fix it!
    for pred_mask, true_mask in zip(pred_masks, true_masks):
        # One label per threshold
        labels = _labeling(pred_mask, true_mask, thresolds)
        jacard_sim_coeff = _custom_precision(labels, total_true_masks, total_pred_masks).mean()
        print(jacard_sim_coeff)
        mean_jacard_sim_coeff += jacard_sim_coeff
    return mean_jacard_sim_coeff / total_true_masks


def _test_dsb_metric():
    """ Check that the metric works as expected: load masks for an image and
    then compute the score comparing these masks with themselves.
    """
    masks = []
    for mask_path in glob.iglob(TEST_MASK_PATH):
        mask = cv2.imread(mask_path)
        masks.append(mask)
    print(dsb_metric(masks, masks))


if __name__ == '__main__':
    _test_dsb_metric()
