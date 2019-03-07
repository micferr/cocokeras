import json

import cv2
import keras
import numpy as np
import matplotlib.image

from pycocotools.coco import COCO
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

'''
    labels = {label: probability}
    Si potrebbe ritornare anche il box migliore

    ATTENZIONE: la dimensione dei boxes deve coincidere con la dimensione
    dell'input della rete

    ATTENZIONE: non c'è nessun controllo sull'output della rete (puo' dare
    in output un vettore più grande (o piccolo) rispetto al numero di categorie
    considerate
'''
class MultiLabelClassificator():
    def __init__(self, model, boxes, preprocess=None, params=None, threshold=0.5):
        self.model = model
        self.boxes = boxes
        self.preprocess = preprocess
        self.params = params
        self.threshold = threshold

    def predict(self, input_img):
        labels = dict()
        for img_size, box_list in self.boxes.items():
            img = cv2.resize(input_img, (img_size, img_size))
            for x1, y1, x2, y2 in box_list:
                subimg = img[y1:y2, x1:x2]

                # si puo' preprocessare direttamente l'immagine intera???
                if self.preprocess != None:
                    subimg = self.preprocess(subimg, self.params)

                predictions = self.model.predict(np.asarray([subimg]))[0]
                best_pred = np.argmax(predictions)
                pred_val  = predictions[best_pred]
                if pred_val >= self.threshold:
                    if best_pred in labels:
                        labels[best_pred] = max(labels[best_pred], pred_val)
                    else:
                        labels[best_pred] = pred_val
        return labels

    def evaluate(self):
        print("Not implemented yet")
        raise
        
'''
    TODO: forse dovrei arrotondare anzichè troncare l'ultima parte di immagine
    res: {input_size1:
              [box_1, box_2, ..],
          input_size2: ...}
    Ciascun box è una tupla di 4 elementi:
        x1 -> coordinata x dell'angolo in alto a sinistra
        y1 -> coordinata y dell'angolo in alto a sinistra
        x2 -> coordinata x dell'angolo in bass a destra
        y2 -> coordinata y dell'angolo in bass a destra
'''
def generate_boxes(input_size, box_size, x_step, y_step, scale):
    res = dict()
    while input_size >= box_size:
        res[input_size] = []
        row = 0
        while True:
            col = 0
            y1 = y_step * row
            y2 = y1 + box_size
            if y2 > input_size: break
            while True:
                x1 = x_step * col
                x2 = x1 + box_size
                if x2 > input_size: break
                res[input_size].append((x1, y1, x2, y2))
                col += 1
            row += 1
        input_size //= scale
    return res

