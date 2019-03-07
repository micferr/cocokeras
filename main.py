import os

import cv2
from pycocotools.coco import COCO
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import save_model, load_model

import coco_utils
import inception
from settings import *
from extract_box import get_box_to_cats
from generators import BoxGenerator
from classificator import MultiLabelClassificator, generate_boxes

# imgid = 100
# img = matplotlib.image.imread(get_image_path(imgid))
# labels = classificator.predict(img)

# counter = 0
# minsize = 5
# for img_name in os.listdir(BOX_IMG_DIR):
#     img = cv2.imread(os.path.join(BOX_IMG_DIR, img_name))
#     h, w, _ = img.shape
#     if w < minsize or h < minsize:
#         counter += 1
# print(counter)

img_to_cats = get_box_to_cats(BOX_IMG_DIR)
imgids = np.asarray(list(img_to_cats.keys()))
xtrain, xvalid, _, _ = train_test_split(imgids, img_to_cats, test_size=0.8)
print("Length: {}".format(len(xtrain)))

# TRAIN
train_data = BoxGenerator(xtrain, BOX_IMG_DIR, img_to_cats, preprocess=inception.preprocess)
valid_data = BoxGenerator(xvalid, BOX_IMG_DIR, img_to_cats, preprocess=inception.preprocess)
model = inception.create()
inception.train(model, train_data, valid_data)
save_model(model, "wout.h5")

# EVALUATE
# data = BoxGenerator(xvalid, BOX_IMG_DIR, img_to_cats, preprocess=inception.preprocess)
# model = load_model("wout.h5")
# model.compile(
#         optimizer='rmsprop',
#         loss='categorical_crossentropy',
#         metrics=['accuracy']) # Questo si toglie al prossimo training
# res = model.evaluate_generator(data, verbose=1)
# print(res)










