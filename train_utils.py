import json
import random

import cv2
import keras
import matplotlib.image
import numpy as np

import app_params
from app_params import SINGLE_CATEGORIES, SINGLE_CATEGORY

IMAGE_SIZE = 256  # Size of the input images
NORMALIZE = True  # Normalize RGB values
BATCH_SIZE = 32
EPOCHS = 5
EARLY_STOP = True
LEARNING_RATE = 0.1
CONV_LAYERS = 4  # Number of Convolution+Pooling layers
CONV_NUM_FILTERS = 32
CONV_FILTER_SIZE = (5, 5)
CONV_POOLING_SIZE = (3, 3)
CONV_STRIDE = 1


class TrainParams:
    def __init__(
            self,
            nn_id=0,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            early_stop=EARLY_STOP,
            image_size=IMAGE_SIZE,

            conv_layers=CONV_LAYERS,
            conv_num_filters=CONV_NUM_FILTERS,
            conv_filter_size=CONV_FILTER_SIZE,
            conv_pooling_size=CONV_POOLING_SIZE,
            conv_stride=CONV_STRIDE,

            base_dir="./out/"
    ):
        self.nn_id = nn_id
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stop = early_stop
        self.image_size = image_size

        self.conv_layers = conv_layers
        self.conv_num_filters = conv_num_filters
        self.conv_filter_size = conv_filter_size
        self.conv_pooling_size = conv_pooling_size
        self.conv_stride = conv_stride

        self.base_dir = base_dir

    def __str__(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


class CocoBatchGenerator(keras.utils.Sequence):
    def __init__(self, imgids, coco_path, params, imgids_to_cats):
        self.img_order = imgids
        self.coco_path = coco_path
        self.params = params
        self.imgids_to_cats = imgids_to_cats

    def __len__(self):
        return int(np.floor(len(self.img_order) / BATCH_SIZE))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.img_order[index * BATCH_SIZE: (index + 1) * BATCH_SIZE]

        # Generate data
        x, y = self.__data_generation(indexes)
        return x, y

    def on_epoch_end(self):
        random.shuffle(self.img_order)

    def __data_generation(self, imgids):
        # Load image files
        input_imgs = [matplotlib.image.imread(
            self.coco_path + '/images/' + app_params.dataType + '/' + ('0' * (12 - len(str(imgid)))) + str(
                imgid) + '.jpg'
        ) for imgid in imgids]

        # Rescale all images to the same size
        input_imgs = [preprocess_image(img, self.params.image_size, NORMALIZE) for img in input_imgs]

        # Convert the batch's X and Y to be fed to the net
        x_train = np.asarray(input_imgs)
        if not SINGLE_CATEGORIES:
            y_train = np.array([self.imgids_to_cats[imgid] for imgid in imgids])
        else:
            y_train = np.array([[self.imgids_to_cats[imgid][SINGLE_CATEGORY]] for imgid in imgids])
        return x_train, y_train


class KFoldCrossValidator:
    def __init__(self, k, data):
        self.k = k
        self.data = data

    def __len__(self):
        return self.k

    def __getitem__(self, item):
        val_size = len(self.data) // self.k
        train_data = self.data[:item * val_size] + self.data[(item + 1) * val_size:]
        val_data = self.data[item * val_size: (item + 1) * val_size]
        return train_data, val_data


def preprocess_image(img, image_size, normalize):
    """
    Returns the result of image preprocessing on img
    """
    # Rescale image to a fixed size
    img = cv2.resize(img, (image_size, image_size))

    # If grayscale, convert to RGB
    if img.shape == (image_size, image_size):
        img = np.repeat(img, 3).reshape(image_size, image_size, 3)

    # If enabled, normalize pixel values (ranges from [0 - 255] to [-1.0 - 1.0])
    if normalize:
        img = ((img / 255.0) - .5) * 2

    return img


def set_random_params(p):
    def make_random_params():
        param_values = {
            'learning_rate': [0.001, 0.01, 0.1, 1.0],
            'conv_layers': [1, 2, 4, 8],
            'conv_num_filters': [16, 32, 64],
            'conv_filter_size': [2, 3, 4, 6, 10],
            'conv_stride': [1, 2, 3, 4, 5],
            'conv_pooling_size': [2, 3, 5, 10]
        }
        res = {}
        for k, v in param_values.items():
            res[k] = random.choice(v)
        return res

    for k, v in make_random_params().items():
        setattr(p, k, v)
