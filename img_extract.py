import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
#from keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import pickle 
from pickle import dump
import shutil 

#from tensorflow.keras.applications.vgg19 import VGG19
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model


img_feat={}

base_model=ResNet50(weights="imagenet")
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)



'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)'''
f_path = 'Images'
images =  os.listdir(f_path)

ctt = 0
ct= 0
for i in images:
    ct+=1
    print("step : ",ct,"completed")
    
    
    img_id=[]
    img_bgr=[]
    
    #img_jpg=f'/Facebook_data/img/modified_Image/{i}.jpg'
    #img_jpeg=f'/Facebook_data/img/modified_Image/{i}.jpeg'
    #img_png=f'/Facebook_data/img/modified_Image/{i}.png'
    
    if '.jpg' in i:
        #img_jpg=np.asarray(Image.open(img_jpg))
        #img_bgr.append(cv2.cvtColor(img_jpg, cv2.COLOR_RGB2BGR))
        img_id.append(i)
        
    if '.jpeg' in i:
        #img_jpeg=np.asarray(Image.open(img_jpeg))
        #img_bgr.append(cv2.cvtColor(img_jpeg, cv2.COLOR_RGB2BGR))
        img_id.append(i)
    
    if '.png' in i:
        #img_png=np.asarray(Image.open(img_png))
        #img_bgr.append(cv2.cvtColor(img_png, cv2.COLOR_RGB2BGR))
        img_id.append(i)
    
        #load image

    
    tt = f_path+i
    image=load_img(tt,target_size=(224,224))
        
    #converting into array
    image=img_to_array(image)
    
    #making it (batch_size,224,224,channels)
    image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    
    #preprocessing the image
    image=preprocess_input(image)
    
    #base_model=VGG19(weights="imagenet")
    
    
    
    #model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    
    #print(model.summary())
    #image = image.to(device)
    feature=model.predict(image)
    
    #img_feat[img_id[0]]=feature
    img_feat[i] = (feature)
    
    #print(img_feat[img_id[0]].shape)
    #print(img_feat)
    #input()
    
    



print("saving data ...")
output=open("images_resnet_gpu_.pkl","wb")
pickle.dump(img_feat,output)

#(batch_size,4096)
