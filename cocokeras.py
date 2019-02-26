import copy
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
NORMALIZE = True  # Normalize RGB values
NORMALIZE_CLASS_WEIGHTS = True
BATCH_SIZE = 32
EPOCHS = 5
EARLY_STOP = True
DO_KFOLD_CROSSVAL = False

LEARNING_RATE = 0.1
CONV_LAYERS = 4  # Number of Convolution+Pooling layers
CONV_NUM_FILTERS = 32
CONV_FILTER_SIZE = (5, 5)
CONV_POOLING_SIZE = (3, 3)
CONV_STRIDE = 1

SINGLE_CATEGORIES = False
SINGLE_CATEGORY = 1

SAVE_MODEL = True

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

weights = np.zeros([NUM_CATEGORIES, 2])
for i in range(NUM_CATEGORIES):
    weights[i][0] = sum([imgids_to_cats[id][i] for id in imgids_to_cats])
    weights[i][1] = len(imgids_to_cats) - weights[i][0]
    if NORMALIZE_CLASS_WEIGHTS:
        weights[i][0] /= len(imgids_to_cats)
        weights[i][1] /= len(imgids_to_cats)
        weights[i] += 0.01  # Account for totally imbalanced classes


class CocoBatchGenerator(keras.utils.Sequence):
    def __init__(self, imgids, coco_path):
        self.img_order = imgids
        self.coco_path = coco_path

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
            self.coco_path + '/images/' + ('0' * (12 - len(str(imgid)))) + str(imgid) + '.jpg'
        ) for imgid in imgids]

        # Rescale all images to the same size
        input_imgs = [cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) for img in input_imgs]

        # Convert grayscale images to RGB
        for index in range(len(input_imgs)):
            if input_imgs[index].shape == (IMAGE_SIZE, IMAGE_SIZE):
                input_imgs[index] = np.repeat(input_imgs[index], 3).reshape(IMAGE_SIZE, IMAGE_SIZE, 3)

        # If enabled, normalize pixel values (ranges from [0 - 255] to [0.0 - 1.0])
        if NORMALIZE:
            input_imgs = [img / 255.0 for img in input_imgs]
            input_imgs = [(img-.5)*2 for img in input_imgs]

        # Convert the batch's X and Y to be fed to the net
        x_train = np.asarray(input_imgs)
        if not SINGLE_CATEGORIES:
            y_train = np.array([imgids_to_cats[imgid] for imgid in imgids])
        else:
            y_train = np.array([[imgids_to_cats[imgid][SINGLE_CATEGORY]] for imgid in imgids])
        return x_train, y_train


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


def weighted_loss(y_true, y_pred):
    '''return -keras.backend.mean(
        weights[:, 0] * (y_true) * keras.backend.log(y_pred) + weights[:, 1] * (1 - y_true) * keras.backend.log(
            1 - y_pred),
        axis=-1)'''
    '''return keras.backend.mean(
        (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** (y_true)) * keras.backend.binary_crossentropy(y_true,
                                                                                                          y_pred),
        axis=-1)'''
    return -keras.backend.mean(
        weights[:, 0] * (1 - y_true) * keras.backend.log(1 - y_pred) +
        weights[:, 1] * (y_true) * keras.backend.log(y_pred),
        axis=-1)


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

    model.compile(
        optimizer=RMSprop(),
        metrics=['accuracy'],
        loss=weighted_loss
    )
    print(model.summary())
    plot_model(model, params.base_dir + 'graph' + str(params.nn_id) + '.png', show_shapes=True)

    train_generator = CocoBatchGenerator(data[0], dataDir)
    val_generator = CocoBatchGenerator(data[1], dataDir)
    callbacks = [TensorBoard(log_dir='./tb')]
    if params.early_stop:
        callbacks += [EarlyStopping('val_loss', patience=2)]

    history = model.fit_generator(
        train_generator,
        epochs=params.epochs,
        callbacks=callbacks,
        validation_data=val_generator
    )

    if SAVE_MODEL:
        save_model(model, params.base_dir + "model" + str(params.nn_id) + '_' + str(kfold_cross_iteration) + ".h5")
    with open(params.base_dir + "history" + str(params.nn_id) + '_' + str(kfold_cross_iteration) + ".txt", "w+") as f:
        f.write('epoch,val_acc,val_loss\n')
        for i in range(len(history.history['val_loss'])):
            f.write("{},{},{}\n".format(i + 1, history.history['val_acc'][i], history.history['val_loss'][i]))

    plot_x = list(range(1, len(history.history['val_loss']) + 1))
    plot_y = history.history['val_loss']

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(0.0, params.epochs)
    plt.ylim(0.0, 1.0)
    plt.plot(plot_x, plot_y, color='blue', linestyle='-')
    plt.savefig(params.base_dir + 'loss' + str(params.nn_id) + '_' + str(kfold_cross_iteration) + '.png', dpi=300)

    return history


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

param_values = {
    'learning_rate': [0.001, 0.01, 0.1, 1.0],
    'conv_layers': [1, 2, 4, 8],
    'conv_num_filters': [16, 32, 64],
    'conv_filter_size': [2, 3, 4, 6, 10],
    'conv_stride': [1, 2, 3, 4, 5],
    'conv_pooling_size': [2, 3, 5, 10]
}


def make_random_params():
    res = {}
    for k, v in param_values.items():
        res[k] = random.choice(v)
    return res


def set_random_params(p, values):
    for k, v in values.items():
        setattr(p, k, v)


random_times = 1
random_results = []
params.nn_id = 0
while params.nn_id != random_times:
    try:
        rand_params = make_random_params()
        set_random_params(params, rand_params)
        params.conv_num_filters = 64
        params.epochs = 10
        params.conv_pooling_size = 2
        params.conv_layers = 2
        params.conv_stride = 2
        params.conv_filter_size = 4
        params.early_stop = False

        with open(params.base_dir + "params" + str(params.nn_id) + ".txt", "w+") as f:
            f.write(str(params))

        val_acc = 0.0
        for k in range(len(kcross)):
            history = train_model(params, kcross[k], k)
            val_acc += history.history['val_acc'][-1]
            if not DO_KFOLD_CROSSVAL:
                val_acc *= len(kcross)
                break
        val_acc /= len(kcross)
        random_results += [(copy.deepcopy(params), val_acc)]
        params.nn_id += 1
    except:
        print("Invalid params!")

random_results.sort(key=(lambda x: x[1]), reverse=True)
to_print = [(r[0].nn_id, r[1]) for r in random_results]
print('ID\tAcc')
for e in to_print:
    print('{}\t{}'.format(e[0], e[1]))

# Test results

model = keras.models.load_model('out/model0_0.h5', custom_objects={'weighted_loss': weighted_loss})
input_imgs = [matplotlib.image.imread(
    dataDir + '/images/' + ('0' * (12 - len(str(imgid)))) + str(imgid) + '.jpg'
) for imgid in imageIds[-200:]]

# Rescale all images to the same size
input_imgs = [cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) for img in input_imgs]

# Convert grayscale images to RGB
for index in range(len(input_imgs)):
    if input_imgs[index].shape == (IMAGE_SIZE, IMAGE_SIZE):
        input_imgs[index] = np.repeat(input_imgs[index], 3).reshape(IMAGE_SIZE, IMAGE_SIZE, 3)

# If enabled, normalize pixel values (ranges from [0 - 255] to [-1.0 - 1.0])
if NORMALIZE:
    input_imgs = [img / 255.0 for img in input_imgs]
    input_imgs = [(img-.5)*2 for img in input_imgs]

# Test x
x = np.asarray(input_imgs)

res = model.predict(x)
pr = []
for i in range(NUM_CATEGORIES):
    tp, fp, tn, fn = 0, 0, 0, 0
    for j in range(len(res)):
        if res[j][i] >= 0.5:
            if imgids_to_cats[imageIds[j]][i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if imgids_to_cats[imageIds[j]][i] == 1:
                fn += 1
            else:
                tn += 1
    print('Category {}'.format(i))
    print('TP\tFP\tTN\tFN ')
    print('{}\t{}\t{}\t{}\tCorrect: {}'.format(tp, fp, tn, fn, (tp + tn) / (tp + fp + tn + fn)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1
    pr += [(precision, recall)]
    fscore = 2 * (precision * recall) / (precision + recall) if precision+recall > 0 else 1
    print('Prec: {}\tRecall: {}\tF1: {}'.format(precision, recall, fscore))
    print('')
