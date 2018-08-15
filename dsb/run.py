import numpy as np
from keras.models import load_model

from dsb.conf import (BATCH_SIZE, EARLY_STOPPING_CALLBACK, EPOCHS,
                      MODEL_CHECKPOINT_CALLBACK, TB_CALLBACK)
from dsb.metric import keras_dsb_metric
from dsb.models.u_net import build_u_net_model
from dsb.postprocessing import prob_to_rles
from dsb.processing import process_test_data, process_train_data

# TODO: Try to use // training and // image processing later.
# TODO: Add mlflow monitoring (useful for trying various hp values).


def ml_pipeline(debug=True):
    """ Get and process the raw images, build a U-Net model, then train it.
    """
    train_images, train_masks = process_train_data(debug)
    assert len(train_images) == len(train_masks), "Something wrong happened!"
    model = build_u_net_model()
    model.fit(train_images, train_masks, callbacks=[TB_CALLBACK,
                                                    MODEL_CHECKPOINT_CALLBACK, EARLY_STOPPING_CALLBACK],
              epochs=EPOCHS, batch_size=BATCH_SIZE)
    return model


def preare_submission(model_path, debug=True):
    """ Use a trained model and test data to prepare the submission file.
    """

    model = load_model(model_path, custom_objects={'keras_dsb_metric': keras_dsb_metric})
    test_images = process_test_data(debug)
    preds_test = model.predict(test_images, verbose=1)

    # Threshold predictions
    preds_test_t = (preds_test > 0.5).astype(np.uint8)

    # TODO: Finish adapting this part.
    # Create list of upsampled test masks
    preds_test_upsampled = []
    for i in range(len(preds_test)):
        preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                           (sizes_test[i][0], sizes_test[i][1]),
                                           mode='constant', preserve_range=True))

    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    # TODO: Add saving part


if __name__ == "__main__":
    ml_pipeline(debug=False)
