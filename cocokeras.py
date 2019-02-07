import json
import random

import cv2
import keras
import matplotlib.image
import numpy as np

import matplotlib.pyplot as plt

from pycocotools.coco import COCO

from keras import Model
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import RMSprop
from keras.utils import plot_model
from keras.models import save_model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, Input, MaxPooling2D

NUM_CATEGORIES = 91  # Total number of categories in Coco dataset
IMAGE_SIZE = 256  # Size of the input images
LEARNING_RATE = 0.1
NORMALIZE = True  # Normalize RGB values
BATCH_SIZE = 32
EPOCHS = 10

CONV_LAYERS = 4  # Number of Convolution+Pooling layers
CONV_NUM_FILTERS = 32
CONV_FILTER_SIZE = (5, 5)
CONV_POOLING_SIZE = (3, 3)
CONV_STRIDE = 1

SINGLE_CATEGORIES = False
SINGLE_CATEGORY = 1

dataDir = 'coco'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

# initialize COCO api for instance annotations
coco = COCO(annFile)

# get all ids
imageIds = [img['id'] for img in coco.dataset['images']]

# Create image_id -> categories mapping
imgids_to_cats = {}
for img in coco.dataset['images']:
    imgid = img['id']
    imgids_to_cats[imgid] = [0] * NUM_CATEGORIES

for ann in coco.dataset['annotations']:
    imgid = ann['image_id']
    catid = ann['category_id']
    imgids_to_cats[imgid][catid - 1] = 1

input_imgids = [img['id'] for img in coco.dataset['images']]


class CocoBatchGenerator(keras.utils.Sequence):
    def __init__(self, imgids):
        self.img_order = imgids

    def __len__(self):
        return int(np.floor(len(self.img_order) / BATCH_SIZE) * 0.1)

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.img_order[index * BATCH_SIZE: (index + 1) * BATCH_SIZE]

        # Generate data
        x, y = self.__data_generation(indexes)
        return x, y

    def on_epoch_end(self):
        random.shuffle(self.img_order)

    def __data_generation(self, _imgids):
        # Load image files
        _input_imgs = [matplotlib.image.imread(
            dataDir + '/images/' + ('0' * (12 - len(str(imgid)))) + str(imgid) + '.jpg'
        ) for imgid in _imgids]

        # Rescale all images to the same size
        _input_imgs = [cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) for img in _input_imgs]

        # Convert grayscale images to RGB
        for _index in range(len(_input_imgs)):
            if _input_imgs[_index].shape == (IMAGE_SIZE, IMAGE_SIZE):
                _input_imgs[_index] = np.repeat(_input_imgs[_index], 3).reshape(IMAGE_SIZE, IMAGE_SIZE, 3)

        # If enabled, normalize pixel values (ranges from [0 - 255] to [0.0 - 1.0])
        if NORMALIZE:
            _input_imgs = [img / 255.0 for img in _input_imgs]

        # Convert the batch's X and Y to be fed to the net
        _x_train = np.asarray(_input_imgs)
        if not SINGLE_CATEGORIES:
            _y_train = np.array([imgids_to_cats[imgid] for imgid in _imgids])
        else:
            _y_train = np.array([[imgids_to_cats[imgid][SINGLE_CATEGORY]] for imgid in _imgids])
        return _x_train, _y_train


class TrainParams:
    def __init__(
            self,
            nn_id=0,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            early_stop=True,
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


def train_model(params, data, kfold_cross_iteration):
    """
    :type params: TrainParams
    """
    # Create the model
    input = Input(shape=(params.image_size, params.image_size, 3))
    for i in range(params.conv_layers):
        x = Conv2D(
            params.conv_num_filters,
            params.conv_filter_size,
            strides=params.conv_stride,
            activation='relu',
            padding='same'
        )(x if i != 0 else input)
        x = MaxPooling2D(pool_size=params.conv_pooling_size)(x)
    x = Flatten()(x)
    out = Dense(NUM_CATEGORIES if (not SINGLE_CATEGORIES) else 1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=out)

    model.compile(optimizer=RMSprop(), loss='binary_crossentropy')
    print(model.summary())
    train_generator = CocoBatchGenerator(data[0])
    val_generator = CocoBatchGenerator(data[1])
    callbacks = []
    if params.early_stop:
        callbacks += [EarlyStopping('loss', patience=2)]
    history = model.fit_generator(
        train_generator,
        epochs=params.epochs,
        callbacks=callbacks,
        validation_data=val_generator
    )

    with open(params.base_dir + "history" + str(params.nn_id) + '_' + str(kfold_cross_iteration) + ".txt", "w+") as f:
        for i in range(len(history.history['val_loss'])):
            f.write("{} {}\n".format(i + 1, history.history['val_loss'][i]))


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


params = TrainParams()
kcross = KFoldCrossValidator(4, input_imgids)
for lr in [0.001, 0.01]:  # Learning rate
    for cl in [1, 2]:  # Num conv layers
        params.learning_rate = lr
        params.conv_layers = cl
        for k in range(len(kcross)):
            train_model(params, kcross[k], k)
        with open(params.base_dir + "params" + str(params.nn_id) + ".txt", "w+") as f:
            f.write(str(params))
        params.nn_id += 1
