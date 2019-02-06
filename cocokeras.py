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
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, Input, MaxPooling2D

NUM_CATEGORIES = 91  # Total number of categories in Coco dataset
IMAGE_SIZE = 256  # Size of the input images
NORMALIZE = True  # Normalize RGB values
BATCH_SIZE = 32
EPOCHS = 100

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
    def __init__(self):
        self.img_order = list(range(len(input_imgids)))
        self.batch_index = 0

    def __len__(self):
        return int(np.floor(len(self.img_order) / BATCH_SIZE) * 0.1)

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.img_order[self.batch_index * BATCH_SIZE: (self.batch_index + 1) * BATCH_SIZE]
        self.batch_index += 1

        # Generate data
        x, y = self.__data_generation(indexes)
        return x, y

    def on_epoch_end(self):
        random.shuffle(self.img_order)
        self.batch_index = 0

    def __data_generation(self, _imgids):
        # Get images' IDs
        _imgids = [coco.dataset['images'][i]['id'] for i in _imgids]

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


class TrainParams():
    def __init__(
            self,
            nn_id="",
            batch_size=BATCH_SIZE,
            epochs=100,
            early_stop=True,

            image_size=IMAGE_SIZE,

            conv_layers=CONV_LAYERS,
            conv_num_filters=CONV_NUM_FILTERS,
            conv_filter_size=CONV_FILTER_SIZE,
            conv_pooling_size=CONV_POOLING_SIZE,
            conv_stride=CONV_STRIDE
    ):
        self.nn_id = nn_id
        self.batch_size = batch_size,
        self.epochs = epochs,
        self.early_stop = early_stop,

        self.image_size = image_size

        self.conv_layers = conv_layers
        self.conv_num_filters = conv_num_filters
        self.conv_filter_size = conv_filter_size
        self.conv_pooling_size = conv_pooling_size
        self.conv_stride = conv_stride


def train_model(params):
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
    plot_model(model, 'model' + str(params.nn_id) + '.png', show_shapes=True)
    training_generator = CocoBatchGenerator()
    history = model.fit_generator(
        training_generator,
        epochs=params.epochs,
        callbacks=[
            TensorBoard(log_dir='./tb')
        ] + ([EarlyStopping('loss', patience=2)] if params.early_stop else [])
    )

    plot_x = list(range(1, len(history.history['loss']) + 1))
    plot_y = np.array(history.history['loss'])

    plt.figure(2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(0.0, 100.0)
    plt.ylim(0.0, 1.0)
    plt.plot(plot_x, plot_y, color='blue', linestyle='-')
    plt.savefig('loss' + str(params.nn_id) + '.png', dpi=300)


params = TrainParams()
train_model(params)
