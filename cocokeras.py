import random

import cv2
import keras
import matplotlib.image
import numpy as np
from keras import Model

from pycocotools.coco import COCO

from keras.utils import plot_model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, Input, MaxPooling2D

NUM_CATEGORIES = 91
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
NORMALIZE = True
BATCH_SIZE = 32
EPOCHS = 12

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
        return int(np.floor(len(self.img_order) / BATCH_SIZE))

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
        _input_imgs = [cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT)) for img in _input_imgs]

        # Convert grayscale images to RGB
        for _index in range(len(_input_imgs)):
            if _input_imgs[_index].shape == (256, 256):
                _input_imgs[_index] = np.repeat(_input_imgs[_index], 3).reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3)

        # If enabled, normalize pixel values (ranges from [0 - 255] to [0.0 - 1.0])
        if NORMALIZE:
            _input_imgs = [img / 255.0 for img in _input_imgs]

        # Convert the batch's X and Y to be fed to the net
        _x_train = np.asarray(_input_imgs)
        _y_train = np.array([imgids_to_cats[imgid] for imgid in _imgids])
        return _x_train, _y_train


# Create the model
input = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
x = Conv2D(32, (5, 5), activation='relu')(input)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(32, (5, 5), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(32, (5, 5), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(32, (5, 5), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
out = Dense(NUM_CATEGORIES, activation='sigmoid')(x)
model = Model(inputs=input, outputs=out)

model.compile(optimizer='rmsprop', loss='binary_crossentropy')
print(model.summary())
plot_model(model, 'model.png', show_shapes=True)
training_generator = CocoBatchGenerator()
model.fit_generator(training_generator, epochs=EPOCHS)
