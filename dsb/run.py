import click
import numpy as np
import pandas as pd
from keras.models import load_model
from skimage.transform import resize

from dsb.conf import (BATCH_SIZE, EARLY_STOPPING_CALLBACK, EPOCHS,
                      MODEL_CHECKPOINT_CALLBACK, TB_CALLBACK, TEST_IMAGE_IDS,
                      VALIDATION_SPLIT)
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
    # Scale train_masks to be in the range [0, 1] (otherwise, loss is <0).
    train_masks /= 255.
    assert len(train_images) == len(train_masks), "Something wrong happened!"
    model = build_u_net_model()
    # TODO: Is the shuffle parameter set to True by default?
    model.fit(train_images, train_masks, validation_split=VALIDATION_SPLIT,
              callbacks=[TB_CALLBACK,
                         MODEL_CHECKPOINT_CALLBACK, EARLY_STOPPING_CALLBACK],
              epochs=EPOCHS, batch_size=BATCH_SIZE,  shuffle=True)
    return model


def preare_submission(model_path, output_path, debug=True):
    """ Use a trained model and test data to prepare the submission file.
    """

    model = load_model(model_path, custom_objects={'keras_dsb_metric': keras_dsb_metric})
    test_images, test_sizes = process_test_data(debug)
    preds_test = model.predict(test_images, verbose=1)
    preds_test *= 255.

    # TODO: Finish adapting this part.
    # Create list of upsampled test masks
    preds_test_upsampled = []
    # TODO: Is it possible to vectorize this loop?
    for i in range(len(preds_test)):
        original_img_shape = test_sizes[i]
        upsampled_img = resize(np.squeeze(preds_test[i]), original_img_shape, mode='constant', preserve_range=True)
        preds_test_upsampled.append(upsampled_img)

    new_test_ids = []
    rles = []
    # run-length encoding.
    # TODO: Improve this step.
    for n, id_ in enumerate(TEST_IMAGE_IDS):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    # Create submission DataFrame
    encoded_pixels_s = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    (pd.DataFrame({"ImageId": new_test_ids, "EncodedPixels": encoded_pixels_s})
        .to_csv(output_path, index=False))

    # Use the kaggle-api to submit.


@click.command()
@click.option('--debug', type=bool,
              default=True, help='Whether to run the pipeline in debug mode or not. Defaults to True.')
def main(debug):
    model = ml_pipeline(debug)
    # TODO: Use logger instead of print
    print(model.summary())


if __name__ == "__main__":
    main()
