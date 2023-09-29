import random
import numpy as np
from utils.data_utils import *
import ast
import albumentations as A
import tensorflow as tf

import pandas as pd


class Dataset:
    def __init__(self,paths,labels,batch_size=4,aug=None):
        self.paths=paths
        self.labels=labels
        self.batch_size=batch_size
        self.num_samples = len(self.labels)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0
        self.aug=aug

    def __iter__(self):
        return self
    def process_data(self,image_path,annotation):
        annot=ast.literal_eval(annotation) #decoding string to dict
        image=cv2.imread(image_path)
        h,w,_=image.shape
        shape=(h,w)
        regions=annot["regions"]
        mask=None
        for region in regions:
            shape_attributes=region["shape_attributes"]
            name=region["region_attributes"]["identity"]
            if name == "odometer":
                all_pointsx=shape_attributes["all_points_x"]
                all_pointsy=shape_attributes["all_points_y"]
                mask=get_segmask((all_pointsx,all_pointsy),shape=shape)# assuming a image has single odometer reading
                break
        return image,mask
    def __next__(self):
        num=0
        
        if self.batch_count<self.num_batchs:
            images=[]
            masks=[]
            
            while num<self.batch_size:
                index=self.batch_count*self.batch_size+num
                try:
                    image_path=self.paths[index]
                    label=self.labels[index]
                except:
                    self.batch_count = 0
                    data=pd.read_csv("train.csv")
                    data=data.sample(frac = 1)
                    self.paths=list(data["paths"].values)
                    self.labels=list(data["labels"].values)
                    raise StopIteration

                image,mask=self.process_data(image_path=image_path,annotation=label)
                if mask is None:
                    num+=1
                    continue
                if self.aug:
                    transforms=self.aug(image=image,mask=mask)
                    image=transforms["image"]
                    mask=transforms["mask"]
                image=image/255.

                image=tf.constant(image, dtype=tf.float32)
                mask=tf.constant(mask, dtype=tf.float32)
                num+=1
                mask=tf.expand_dims(mask,axis=-1)

                images.append(image)
                masks.append(mask)

            image_batch=tf.stack(images,axis=0)
            masks_batch=tf.stack(masks,axis=0)
            self.batch_count+=1
            return(image_batch,masks_batch)
        else:
            self.batch_count = 0
            data=pd.read_csv("train.csv")
            data=data.sample(frac = 1)
            self.paths=list(data["paths"].values)
            self.labels=list(data["labels"].values)
            raise StopIteration
        
    def __len__(self):
        return self.num_batchs







                




        
        