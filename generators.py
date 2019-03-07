import os
import random

import keras
import matplotlib
import numpy as np

import coco_utils
from settings import *

'''
    img_to_cats dizionario del tipo
    { imgid1: [cat1, cat2, ..], .. }
    La lista di categorie non è esaustiva questo significa che
    sono presenti solo le categorie alle quali appartiene l'immagine
    rappresentata da imgid
'''
class BaseGenerator(keras.utils.Sequence):
    def __init__(self, imgids, path, img_to_cats, preprocess=None, params=None):
        self.img_order = imgids
        random.shuffle(self.img_order)
        self.path = path
        self.img_to_cats = img_to_cats
        self.preprocess = preprocess
        self.params = params

    def __len__(self):
        return int(np.floor(len(self.img_order) / BATCH_SIZE))

    def __getitem__(self, index):
        s = index * BATCH_SIZE
        e = (index + 1) * BATCH_SIZE
        imgids = self.img_order[s:e]
        return self.__data_generation(imgids)

    def on_epoch_end(self):
        random.shuffle(self.img_order)

    def __data_generation(self, imgids):
        input_imgs = [matplotlib.image.imread(
            os.path.join(self.path, self._get_img_name(imgid))) for imgid in imgids]

        if self.preprocess != None:
            input_imgs = [self.preprocess(img, self.params) for img in input_imgs]

        xset = np.asarray(input_imgs)
        yset = np.array([self._get_img_cats(imgid) for imgid in imgids])
        return xset, yset

    def _get_img_name(self, imgid):
        print("_get_img_name should be overrided")
        raise

    def _get_img_cats(self, imgid):
        print("_get_img_cats should be overrided")
        raise

class CocoGenerator(BaseGenerator):
    def __init__(self, imgids, path, img_to_cats, preprocess=None, params=None):
        super(CocoGenerator, self).__init__(imgids, path, img_to_cats, preprocess, params)

    def _get_img_name(self, imgid):
        return coco_utils.get_img_name(imgid)

    def _get_img_cats(self, imgid):
        res = np.zeros(NUM_CATEGORIES)
        for catid in self.img_to_cats[imgid]:
            res[catid] = 1
        return res

class BoxGenerator(BaseGenerator):
    def __init__(self, imgids, path, img_to_cats, preprocess=None, params=None):
        super(BoxGenerator, self).__init__(imgids, path, img_to_cats, preprocess, params)

    def _get_img_name(self, imgid):
        return str(imgid) + '_' + str(self.img_to_cats[imgid]) + '.jpg'

    def _get_img_cats(self, imgid):
        res = np.zeros(NUM_CATEGORIES)
        cat = self.img_to_cats[imgid]
        res[cat] = 1
        return res

