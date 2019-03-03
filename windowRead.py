# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 17:34:10 2019

@author: Monster
"""

import scipy.io
import numpy as np
from trainFeature import preprocessImage
import resnet
from PIL import Image
from tqdm import tqdm
import torch
#import h5py


if __name__ == '__main__':
    mat = scipy.io.loadmat('wind.mat')
    n = mat["Allimwindows"]
    model = resnet.resnet50(pretrained = True)
    #model.eval()
    mat = 0
    features = []
    
    for i in tqdm(range(100)):    
        image_features = []
        
        for j in tqdm(range(50)):
            arr = np.array(n[0][i][0][j])#0 testImage 0 window
            arrImage = Image.fromarray(arr, 'RGB')
            new_image = preprocessImage(arrImage)
            new_image = np.reshape(new_image, [1, 224, 224, 3])
            new_image = np.transpose(new_image, [0, 3, 1, 2])
                    
            new_image = torch.from_numpy(new_image)
            
            feature_vector = model(new_image)
            feature_vector = feature_vector.detach().numpy()
            image_features.append(feature_vector)
        #features.append(image_features)
        scipy.io.savemat("test_features/"+str(i)+".mat", mdict={'feature':image_features}) 
    #features = np.array(features)1
