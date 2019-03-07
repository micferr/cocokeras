import numpy as np
import cv2
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.applications import inception_v3
from keras.optimizers import SGD

from settings import *

# TODO: la size non dovrebbe essere hardcoded
def preprocess(img, params):
    img = cv2.resize(img, (299, 299))
    if img.shape == (299, 299):
        img = np.repeat(img, 3).reshape(299, 299, 3)
    img = ((img / 255) - 0.5) * 2
    return img

# TODO: non mi cago callbacks
# TODO: aggiungere parametri per training
def train(model, train_data, valid_data):
    # training 1
    for layer in model.layers[:311]: # inceptionv3 ha 311 livelli
        layer.trainable = False
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    model.fit_generator(
        train_data,
        epochs=1)

    # training 2
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    model.fit_generator(
        train_data,
        epochs=1)

def create(imagenet_weights=True):
    weights = 'imagenet' if imagenet_weights else None
    inception = inception_v3.InceptionV3(include_top=False, weights=weights,
        input_shape=(299, 299, 3))
    x = inception.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CATEGORIES, activation='softmax')(x)
    model = Model(inputs=inception.input, outputs=predictions)
    return model

