# Competition metric implementation
# Check this Wikipedia page for more details about Jacard index:
# https://en.wikipedia.org/wiki/Jaccard_index


THRESHOLDS = range(0.5, 1, 0.05)


def _custom_precision(labels, total_true, total_pred, positive_class=1):
    """ A custom precision (i.e. different to the common one) metric for a two
    classes (positive and negative) classification.
    Notice that this custom precision is exactly the Jaccard similarity coefficient.
    """
    true_positive = (labels == positive_class).sum()
    false_positive = total_pred - true_positive
    false_negative = total_true - true_positive
    return true_positive / (true_positive + false_positive + false_negative)


def _iou(img_1, img_2):
    """ IoU (intersection over union also called Jacard index) computation for two
    images. Notice that the pixels are transfored into binary values: either bigger than 0 (image) or not
    (background).
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
    for pred_mask, true_mask in zip(pred_masks, true_masks):
        labels = _labeling(pred_mask, true_mask, thresolds)
        jacard_sim_coeff = _custom_precision(labels, total_true_masks, total_pred_masks).mean()
        mean_jacard_sim_coeff += jacard_sim_coeff
    return mean_jacard_sim_coeff / total_true_masks

# TODO: Add function that finds true and predicted masks.


def _test_dsb_metric():
    """ Check that the metric works as expected.
    """
    # TODO: Finish this test.
    assert True
