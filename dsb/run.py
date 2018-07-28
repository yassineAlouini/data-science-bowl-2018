from dsb.models.u_net import build_u_net_model
from dsb.processing import process_images

# TODO: Try to use // training and // image processing later.


def ml_pipeline():
    """ Get and process the raw images, build a U-Net model, then train it.
    """
    imgs = process_images()
    model = build_u_net_model()
    model.fit(imgs)
    return model


if __name__ == "__main__":
    ml_pipeline()
