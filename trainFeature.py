# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import torch
from PIL import Image
import os, os.path
import resnet
from tqdm import tqdm
import scipy.io as sp

def preprocessImage(image):
    x, y = image.size
            
    maxdim = max(x, y)
    new_image = Image.new('RGB', (maxdim, maxdim), (0, 0, 0, 0))
    new_image.paste(image, ((maxdim - x) // 2, (maxdim - y) // 2))
    new_image = new_image.resize((224, 224))
    new_image = np.float32(new_image)
    new_image = np.asarray(new_image)
    for i in range(224):
        for j in range(224):
            pixel = new_image[i, j, :]
            pixel = pixel / 255
            
            pixel[0] = pixel[0] - 0.485
            pixel[0] = pixel[0] / 0.229
            
            pixel[1] = pixel[1] - 0.456
            pixel[1] = pixel[1] / 0.224
            
            pixel[2] = pixel[2] - 0.406
            pixel[2] = pixel[2] / 0.225
            
            new_image[i, j, :] = pixel
    
    return new_image

if __name__ == '__main__':
    TRAIN_DIR = "train"
    
    model=resnet.resnet50(pretrained = True)
    size = 224, 224
    feature_array=[]
    label = []
    currentLabel = 0
    for f in os.listdir(TRAIN_DIR): #assuming gif
        currentLabel = currentLabel + 1
        p = os.path.join(TRAIN_DIR, f)
        for q in tqdm(os.listdir(p)):
            
            label.append(currentLabel)
            filename = os.path.join(p, q)
            
            image=Image.open(filename).convert('RGB')
            
            new_image = preprocessImage(image)
            
            new_image = np.reshape(new_image, [1, 224, 224, 3])
            new_image = np.transpose(new_image, [0, 3, 1, 2])
            
            new_image = torch.from_numpy(new_image)
            
            feature_vector = model(new_image) 
            feature_vector = feature_vector.detach().numpy()
            feature_array.append(feature_vector)
    
    print("Feature extraction finished...")
    feature_array = np.array(feature_array)
    sp.savemat("feature.mat", mdict={'feature_array': feature_array})
    
    label = np.array(label)
    
    sp.savemat("label.mat", mdict={'label':label})
       
    print("End program")

    
    