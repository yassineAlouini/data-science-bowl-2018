""" Some useful constants
"""

import os

import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# TODO: Use Pathlib instead of os.path.
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGES_BASE_PATH = os.path.join(BASE_PATH, 'data')
TRAIN_LABELS_PATH = os.path.join(BASE_PATH, 'data/stage1_train_labels.csv')
ALL_IMAGE_IDS = set(next(os.walk(os.path.join(BASE_PATH, 'data')))[1])
TRAIN_IMAGE_IDS = set(pd.read_csv(TRAIN_LABELS_PATH).ImageId.unique())
TEST_IMAGE_IDS = ALL_IMAGE_IDS - TRAIN_IMAGE_IDS
TEST_MASK_PATH = os.path.join(
    BASE_PATH, "data/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9/masks/*.png")


IMG_CHANNELS = 3
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMAGE_DATA_FORMAT = "channels_last"
# TODO: Use Pathlib instead of os.path.
# Make sure that the TB logs are logged somewhere where you have the rights.
TB_LOG_DIR = os.path.join(BASE_PATH, 'tb_logs')
MODEL_CHECKPOINT_PATH = os.path.join(BASE_PATH, 'models', 'model_dsb.h5')
TB_CALLBACK = TensorBoard(log_dir=TB_LOG_DIR, histogram_freq=0,
                          write_graph=True, write_images=True)
# By default monitors "val_loss" metric (thus need a validation set while fitting the model).
EARLY_STOPPING_CALLBACK = EarlyStopping(patience=20, verbose=1)
MODEL_CHECKPOINT_CALLBACK = ModelCheckpoint(MODEL_CHECKPOINT_PATH, verbose=1, save_best_only=True)
# Some of the model hyperparameters. Tune later.
EPOCHS = 1000
BATCH_SIZE = 8
# Validation size * VALIDATION_SPLIT + Train size * (1 - VALIDATION_SPLIT) = all train size.
VALIDATION_SPLIT = 0.1
