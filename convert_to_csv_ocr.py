import os
import cv2
import json
from utils.data_utils import get_segmask
from utils.inference_helpers import store_crops_ocr
import pandas as pd
root_path="/home/chanukya/Downloads/ClearCut/train"
folders=os.listdir(root_path)

image_paths=[]
annotations=[]
count=0
for folder in folders:
    files=os.listdir(os.path.join(root_path,folder))
    annot_file=json.load(open(root_path+"/"+folder+"/"+"via_region_data.json","r"))
    for file in files:
        if ".json" not in file:
            try :
                image_path=root_path+"/"+folder+"/"+file
                image=cv2.imread(image_path)
                h,w,_=image.shape
                shape=(h,w)
                annotation=annot_file[file]
                regions=annotation["regions"]
                mask=None
                flag=0
                print(regions)
                for region in regions:
                    shape_attributes=region["shape_attributes"]
                    name=region["region_attributes"]["identity"]
                    if name == "odometer":
                        all_pointsx=shape_attributes["all_points_x"]
                        all_pointsy=shape_attributes["all_points_y"]
                        mask=get_segmask((all_pointsx,all_pointsy),shape=shape)
                        path=os.path.join("ocr_train",str(count)+".jpg")
                        store_crops_ocr(original_image=image,mask=mask,path=path)
                        reading=region["region_attributes"]['reading']
                        count+=1
                        flag=1
                if flag:
                    image_paths.append(path)
                    annotations.append(reading)
            except:
                pass



data=pd.DataFrame()

data["paths"]=image_paths
data["labels"]=annotations

data.to_csv("train_ocr.csv",index=False)