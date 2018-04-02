import os

IMAGES_PATH = 'data/**/images/*.png'
TRAIN_LABELS_PATH = 'data/stage1_train_labels.csv'
ALL_IMAGE_IDS = set(next(os.walk('data'))[1])

train_labels_df = pd.read_csv(TRAIN_LABELS_PATH)
TRAIN_IMAGE_IDS = set(train_labels_df.ImageId.unique())
TEST_IMAGE_IDS = ALL_IMAGE_IDS - TRAIN_IMAGE_IDS
