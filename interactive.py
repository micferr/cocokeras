import os

from keras.models import load_model
from pycocotools.coco import COCO
from matplotlib.image import imread

from classificator import MultiLabelClassificator, generate_boxes
from settings import *
import coco_utils
import inception

if __name__ == '__main__':
    img_to_cats = coco_utils.get_img_to_cats(COCO(coco_utils.val_ann))
    model = load_model("wout.h5")
    boxes = generate_boxes(299*2, 299, 149, 149, 2)
    classificator = MultiLabelClassificator(model, boxes, inception.preprocess)
    while 1:
        imgid    = int(input("Insert image id: "))
        imgname  = coco_utils.get_img_name(imgid)
        pathname = os.path.join(COCO_IMG_DIR, imgname)
        if not os.path.exists(pathname):
            print("\nImage {} does not exists\n".format(pathname))
            print("\n----------------------------------------\n")
            continue
        img = imread(pathname)
        predictions = classificator.predict(img)
        print("\nTruth:")
        for c in img_to_cats[imgid]:
            label = coco_utils.decode_label(c)
            print("    - {}".format(label))
        print("\nPredictions:")
        for l, v in predictions.items():
            label = coco_utils.decode_label(l)
            print("    - {}: {}".format(label, v))
        print("\n----------------------------------------\n")

