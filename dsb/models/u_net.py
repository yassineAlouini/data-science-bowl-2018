""" Unet model implementation inspired from this repo:
https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
"""

from __future__ import print_function

from keras import backend as K
from keras.layers import (Conv2D, Conv2DTranspose, Input, Lambda, MaxPooling2D,
                          concatenate)
from keras.models import Model

from dsb.conf import IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH
from dsb.metric import keras_dsb_metric

K.set_image_data_format('channels_last')  # TF dimension ordering in this code


def build_u_net_model():
    """ Unet model
    """
    # TODO: Refactor some of the conv_x layers into a function.
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    scale = Lambda(lambda x: x / 255)(inputs)

    conv_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(scale)
    conv_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv_1)
    pooling_1 = MaxPooling2D((2, 2))(conv_1)

    conv_2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pooling_1)
    conv_2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv_2)
    pooling_2 = MaxPooling2D((2, 2))(conv_2)

    conv_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pooling_2)
    conv_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_3)
    pooling_3 = MaxPooling2D((2, 2))(conv_3)

    conv_4 = Conv2D(64, (3, 3), activation='relu', padding='same')(pooling_3)
    conv_4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_4)
    pooling_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

    conv_5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pooling_4)
    conv_5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_5)

    u_6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv_5)
    u_6 = concatenate([u_6, conv_4])
    conv_6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u_6)
    conv_6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_6)

    u_7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv_6)
    u_7 = concatenate([u_7, conv_3])
    conv_7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u_7)
    conv_7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_7)

    u_8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv_7)
    u_8 = concatenate([u_8, conv_2])
    conv_8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u_8)
    conv_8 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv_8)

    u_9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(conv_8)
    u_9 = concatenate([u_9, conv_1], axis=3)
    conv_9 = Conv2D(8, (3, 3), activation='relu', padding='same')(u_9)
    conv_9 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv_9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv_9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras_dsb_metric])
    return model
