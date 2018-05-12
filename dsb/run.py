from dsb.models.u_net import build_u_net_model
from dsb.processing import process_images


def ml_pipeline():
    # TODO: Add documentation
    imgs = process_images()
    model = build_u_net_model()
    model.fit(imgs)
