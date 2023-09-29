from model.ocr_model import CTCLayer,build_model
from tensorflow import keras
import cv2
import tensorflow as tf
import numpy as np

import json
with open("ocrdict.json","r") as f:
    vocab=json.load(f)
from tensorflow.keras import layers
model=build_model()
model.load_weights("./ocr_wts/ocr_weights_47_1.3271915912628174.h5")

prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
num_to_char = layers.StringLookup(
    vocabulary=vocab, mask_token=None, invert=True
)
class OcrInfernce:
    def __init__(self,):
        self.model=prediction_model

    def preprocess(self,crop):
        gray_image=cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray_image=cv2.resize(gray_image,(128,32))
        gray_image=tf.constant(gray_image,dtype=tf.float32)
        img = tf.transpose(gray_image, perm=[1, 0])
        img_data=img/255.
        img_data=tf.expand_dims(img_data,axis=-1)#assuming only one crop is passed
        img_data=tf.expand_dims(img_data,axis=0)
        return img_data
    
    def decode_batch_predictions(self,pred,max_length=8):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :max_length
        ]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    def predict(self,crop):
        crop=self.preprocess(crop)
        preds=self.model(crop)
        text=self.decode_batch_predictions(preds)

        return text

        
