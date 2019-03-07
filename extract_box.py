import os

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2

from pycocotools.coco import COCO

from settings import *
from coco_utils import *

def get_box_to_cats(path):
    res = dict()
    for img_name in os.listdir(path):
        imgid, catid = img_name.split('.')[0].split('_')
        res[int(imgid)] = int(catid)
    return res

def extract_boxes(dst, src, ann_file):
    coco = COCO(ann_file)
    imgids = [img['id'] for img in coco.dataset['images']]
    count=0
    for ann in coco.dataset['annotations']:
        bb    = ann['bbox']
        catid = ann['category_id'] - 1
        imgid = ann['image_id']
        img = cv2.imread(os.path.join(src, get_img_name(imgid)))
        subimg =  img[int(bb[1]):int(bb[1])+int(bb[3]), int(bb[0]):int(bb[0])+int(bb[2])]
        savepath = os.path.join(dst, str(count) + '_' + str(catid) + '.jpg')
        if not cv2.imwrite(savepath, subimg):
            if os.path.exists(savepath):
                os.remove(savepath)
            print("Failed to save {}".format(savepath))
        count +=1

if __name__ == '__main__':
    extract_boxes(BOX_IMG_DIR, COCO_IMG_DIR, val_ann)
