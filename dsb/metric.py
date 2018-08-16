# Competition's metric implementation
# Check this Wikipedia page for more details about Jacard index:
# https://en.wikipedia.org/wiki/Jaccard_index

import glob

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.losses import binary_crossentropy
from skimage.io import imread

from dsb.conf import TEST_MASK_PATH

START = 0.5
END = 0.95
STEP = 0.05
N_STEPS = int((END - START) / STEP) + 2
THRESHOLDS = np.linspace(START, END, N_STEPS)


def _custom_precision(labels_matrix, total_true, total_pred, positive_class=1):
    """ A custom precision (i.e. different from the common one) metric for a two
    classes (positive and negative) classification.
    Notice that this custom precision is exactly the Jaccard similarity coefficient.
    """
    true_positive = (labels_matrix == positive_class).sum(axis=0)
    false_positive = total_pred - true_positive
    false_negative = total_true - true_positive
    return true_positive / (true_positive + false_positive + false_negative)


def _iou(img_1, img_2):
    """ IoU (intersection over union also called Jacard index) computation for two
    images. Notice that the pixels are first transfmored into binary values: either bigger than 0 (image) or equal to
    0 (background). Notice that 0 denotes the black color.
    """
    # int casting so that numpy bitwise_and works.
    img_1 = np.uint64(img_1)
    img_2 = np.uint64(img_2)
    intersection = (img_1 * img_2).sum()
    union = img_1.sum() + img_2.sum()
    # To avoid NaNs
    if union == 0.0:
        return 0.0
    return intersection / union


def _labeling(pred_mask, true_mask, thresholds):
    """ Label the predicted mask in comparaison to the true one using the IoU (intersection over union) metric
    and a given thresolds (this will broadcast the operation): 1 if IoU > thresholds, 0 otherwise.
    """
    computed_iou = _iou(pred_mask, true_mask)
    return np.uint64(computed_iou > thresholds)


def dsb_metric(true_masks, pred_masks, thresholds=THRESHOLDS):
    """ A specific IoU (intersection over union) metric for the dsb competition
    """
    total_true_masks = len(pred_masks)
    total_pred_masks = len(true_masks)
    labels = [_labeling(pred_mask, true_mask, thresholds) for pred_mask, true_mask in zip(pred_masks, true_masks)]
    labels_matrix = np.matrix(labels)
    assert labels_matrix.shape == (total_true_masks, len(thresholds))  #  (Number of masks, Number of thresholds)
    # A vector of Jaccard similarity index, one per threshold value.
    jacard_sim_coeff = _custom_precision(labels_matrix, total_true_masks, total_pred_masks)
    assert jacard_sim_coeff.shape == (1, len(thresholds))  #  (1, Number of thresholds)
    return jacard_sim_coeff.mean()


def keras_dsb_metric(true_masks, pred_masks):
    """ TF wrapper for the dsb_metric so that it works with Keras.
    """
    return tf.py_func(dsb_metric, [true_masks, pred_masks], tf.float64)


def _test_dsb_metric():
    """ Check that the metric works as expected: load masks for an image and
    then compute the score. For the the original masks, the score should be 1.0. For "background" masks
    (predict a very small value everywhere, 1 for example), the score should be 0.0.
    """
    masks = []
    for mask_path in glob.iglob(TEST_MASK_PATH):
        mask = imread(mask_path)
        masks.append(mask)
    background_mask = np.ones(mask.shape, dtype=np.int)
    background_masks = [background_mask for _ in range(len(masks))]
    assert dsb_metric(masks, masks) == 1.0
    assert dsb_metric(masks, background_masks) == 0.0

# A variant loss to optimize. Inspired from here: https://www.kaggle.com/kmader/baseline-u-net-model-part-1


def dice_coef(y_true, y_pred, smooth=1):
    # TODO: Add some documentation
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce_loss(in_gt, in_pred):
    # TODO: Add some documentation
    return 1e-3 * binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)


if __name__ == '__main__':
    _test_dsb_metric()
