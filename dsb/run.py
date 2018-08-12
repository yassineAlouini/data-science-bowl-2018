from dsb.models.u_net import build_u_net_model
from dsb.processing import process_train_data
from dsb.utils import TB_CALLBACK

# TODO: Try to use // training and // image processing later.


def ml_pipeline():
    """ Get and process the raw images, build a U-Net model, then train it.
    """
    train_images, train_masks = process_train_data()
    assert len(train_images) == len(train_masks), "Something wrong happened!"
    model = build_u_net_model()
    model.fit(train_images, train_masks, callbacks=[TB_CALLBACK])
    return model


if __name__ == "__main__":
    ml_pipeline()
