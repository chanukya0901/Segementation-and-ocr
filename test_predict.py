import tensorflow as tf
from ocr_infer import OcrInfernce
import cv2
import os
from model.seg_model import Segmodel
import albumentations as A
import numpy as np
from copy import deepcopy
from utils.inference_helpers import store_crops
from absl import flags,app
from absl.flags import FLAGS
import pandas as pd
model=Segmodel() ## initiate segmenttaion model
sample_input=tf.random.normal(shape=(4,416,416,3))
model(sample_input) #random pass to load weights
model.load_weights("./wts/wts_0.18531334400177002.h5")
transform=A.Resize(height=416, width=416)
flags.DEFINE_string("folder_path","./test_folder","path to test folder")
text_extractor=OcrInfernce()
def main(_argv):
    def read_odometer(image_path,thresh=0.5):
        image=cv2.imread(image_path)
        original_image=deepcopy(image)
        original_resolution=original_image.shape
        x_scale = original_resolution[1] / 416 #scale of x to extract location on orginal image
        y_scale = original_resolution[0] / 416 #scale of y to extract location on orginal image
        aug=transform(image=image)
        image_final=aug["image"]
        image=image_final/255.
        try:
            image=tf.constant(image, dtype=tf.float32)
            imag_data=tf.expand_dims(image,axis=0)  
            pred=model(imag_data)
            bool_mask=pred>thresh  # converting logits to bool  tensor
            mask=tf.cast(bool_mask,dtype=tf.float32)
            mask=mask.numpy()*255.
            mask=np.array(mask[0,:,:,0],dtype=np.uint8)
            reading_crop_path=store_crops(original_image=original_image,mask=mask,x_scale=x_scale,y_scale=y_scale)
            reading_crop=cv2.imread(reading_crop_path)
            text_pedictions=text_extractor.predict(reading_crop)
            text=text_pedictions[0].replace("*","")#assuming only one crop
            text=text.replace("[UNK]","")
            os.remove(reading_crop_path)
        except :
            
            text="either image may not contain odometer reading or segementation model failed to detect the reading"
        #remove stored patch or crop
        
        return text
    test_folder_path=FLAGS.folder_path
    image_names=os.listdir(test_folder_path)
    imageNames=[]
    texts=[]
    for image_name in image_names:
        if ".json" not in image_name:
            image_path=os.path.join(test_folder_path,image_name)
            text=read_odometer(image_path=image_path)
            imageNames.append(image_name)
            texts.append(text)

    result_data=pd.DataFrame()
    result_data["image_name"]=imageNames
    result_data["predictions"]=texts
    result_data.to_csv("test_result.csv",index=False)
    print("written")

if __name__ == '__main__':
    try:
        app.run(main)
    except Exception as e:
        print(e)


