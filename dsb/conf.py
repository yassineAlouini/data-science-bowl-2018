""" Some useful constants
"""

import os

import pandas as pd

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGES_PATH = os.path.join(BASE_PATH, 'data/**/images/*.png')
TRAIN_LABELS_PATH = os.path.join(BASE_PATH, 'data/stage1_train_labels.csv')
ALL_IMAGE_IDS = set(next(os.walk(os.path.join(BASE_PATH, 'data')))[1])
TRAIN_IMAGE_IDS = set(pd.read_csv(TRAIN_LABELS_PATH).ImageId.unique())
TEST_IMAGE_IDS = ALL_IMAGE_IDS - TRAIN_IMAGE_IDS
TEST_MASK_PATH = os.path.join(
    BASE_PATH, "data/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9/masks/*.png")


IMG_CHANNELS = 3
IMG_WIDTH = 256
IMG_HEIGHT = 256
