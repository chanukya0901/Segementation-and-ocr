import os
import cv2
import json
import pandas as pd

root_path="./train"
folders=os.listdir(root_path)

image_paths=[]
annotations=[]
for folder in folders:
    files=os.listdir(os.path.join(root_path,folder))
    annot_file=json.load(open(root_path+"/"+folder+"/"+"via_region_data.json","r"))
    for file in files:
        if ".json" not in file:
            try :
                image_path=root_path+"/"+folder+"/"+file
                annotation=annot_file[file]
                regions=annotation["regions"]
                mask=None
                flag=0
                print(regions)
                for region in regions:
                    shape_attributes=region["shape_attributes"]
                    name=region["region_attributes"]["identity"]
                    if name == "odometer":
                        flag=1
                if flag:
                    image_paths.append(image_path)
                    annotations.append(annotation)
            except:
                pass

data_total=pd.DataFrame()
data_total["paths"]=image_paths
data_total["labels"]=annotations
print("total data points are :",len(data_total))

# data_total.to_csv("train.csv",index=False)
data=pd.read_csv("train.csv")
print("len of data is",len(data))







