# Competition metric implementation
# Check this Wikipedia page for more details about Jacard index:
# https://en.wikipedia.org/wiki/Jaccard_index


THRESHOLDS = range(0.5, 1, 0.05)


def _custom_precision(y_pred, y_true, positive_class=1):
    """ A custom precision (i.e. different to the common one) metric for a two
    classes (positive and negative) classification.
    Notice that this custom precision is exactly the Jaccard similarity coefficient.
    """
    true_positive = (y_pred == y_true) & (y_pred == positive_class)
    false_positive = (y_pred != y_true) & (y_pred == positive_class)
    false_negative = (y_pred != y_true) & (y_pred == 1 - positive_class)
    return true_positive / (true_positive + false_positive + false_negative)


def _iou(img_1, img_2):
    """ IoU (intersection over union also called Jacard index) computation for two
    images.
    """
    intersection = ((img_1 & img_2) > 0).sum()
    union = ((img_2 | img_2) > 0).sum()
    return intersection / union


def _labeling(pred_mask, true_mask, threshold):
    """ Label the predicted mask in comparaison
    of the true one using the IoU (intersection over union) metric.
    """
    computed_iou = _iou(pred_mask, true_mask)
    return computed_iou > threshold


def dsb_metric(y_pred, y_true, thresolds=THRESHOLDS):
    """ A specific IoU (intersection over union) for the dsb competition
    """
    for threshold in thresholds:
        pass


# TODO: Add test for the metric
def _test_dsb_iou():
    assert True
