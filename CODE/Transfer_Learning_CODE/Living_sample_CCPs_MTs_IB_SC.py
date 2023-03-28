#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:09:02 2021
@author: lhjin
"""

import os
import math
import torch
from torch.utils.data import  DataLoader
from skimage import io, transform
import numpy as np

from unet_model import UNet
import warnings
warnings.filterwarnings('ignore')

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data_in       = sample['image_in']
        name          = sample['image_name']
        max_intensity = sample['max_intensity']
        
        return {'image_in': torch.from_numpy(data_in),
                'image_name':name,
                'max_intensity':max_intensity}

class ReconsDataset(torch.utils.data.Dataset):
     def __init__(self, all_data_path, transform):
        self.all_data_path = all_data_path
        self.transform = transform
        self.dirs_data = os.listdir(self.all_data_path)
     def __len__(self):
         dirs = self.dirs_data
         return len(dirs)          

     def __getitem__(self, idx):
         dirs = self.dirs_data 
         
         file_name = os.path.join(self.all_data_path, dirs[idx])
         data_in = io.imread(file_name)
         
         max_intensity = max(data_in.flatten())
         data_in = data_in/max_intensity
         
         sample = {'image_in': data_in, 'image_name':dirs[idx][:-4],'max_intensity':max_intensity/1.0}
         
         if self.transform:
             sample = self.transform(sample)
         return sample


if __name__ == "__main__":
    cuda = torch.device('cuda:0')
    batch_size = 1
    input_size= 1
    # output_size = 1
    SRRFDATASET = ReconsDataset(all_data_path="/home/jlh/20220916/0-SIM-YongXin/CCPs_MTs/",
                                transform = ToTensor())   
   
    test_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size, shuffle=False, pin_memory=True) # better than for loop
    model_1 = UNet(n_channels=input_size, n_classes=1)
    model_1.cuda(cuda)
    model_1.load_state_dict(torch.load("/home/jlh/20220916/Models/YXSIM_transfer/IB/Models_4540.pkl"))
    model_1.eval()
    
    model_2 = UNet(n_channels=input_size, n_classes=2)
    model_2.cuda(cuda)
    model_2.load_state_dict(torch.load("/home/jlh/20220916/Models/YXSIM_transfer/SC-2/Models_3330.pkl"))
    model_2.eval()
    
    f_p_s = '/home/jlh/20220916/Prediction/SIM_YongXin_testing/YXSIM_transfer/CCPs_MTs_crop/'
    file_path = f_p_s + 'CCPs_MTs_IB/'
    folder = os.path.exists(file_path)
    if not folder:
        os.makedirs(file_path)
    file_path_2 = f_p_s + 'CCPs_MTs_IB_SC/'
    folder = os.path.exists(file_path_2)
    if not folder:
        os.makedirs(file_path_2)
        
    for batch_idx, items in enumerate(test_dataloader):
        image         = items['image_in']
        image_name    = items['image_name']
        max_intensity = items['max_intensity']
        image = image.unsqueeze(dim=3)
        image = np.swapaxes(image, 1,3)
        image = np.swapaxes(image, 2,3)
        image = image.float()
        image = image.cuda(cuda)          
        pred = model_1(image)
        
        max_intensity = max_intensity.float()
        max_intensity = max_intensity.cuda(cuda)  
        
        I = pred[0,0,:,:]
        mv = I.flatten().min()
        if mv < 0:
            I = I + abs(mv)
        I = I*max_intensity
        prediction_save_path = file_path+image_name[0]+'.tif'
        io.imsave(prediction_save_path, I.detach().cpu().numpy().astype(np.uint16))
        
       
        image = pred;
        pred = model_2(image)
        
        I = pred[0,0,:,:]
        mv = I.flatten().min()
        if mv < 0:
            I = I + abs(mv)        
        I = I*max_intensity
        prediction_save_path  = file_path_2+image_name[0]+'_CCPs.tif'
        io.imsave(prediction_save_path , I.detach().cpu().numpy().astype(np.uint16))
        
        I = pred[0,1,:,:]
        mv = I.flatten().min()
        if mv < 0:
            I = I + abs(mv)
        I = I*max_intensity
        prediction_save_path  = file_path_2+image_name[0]+'_MTs.tif'
        io.imsave(prediction_save_path , I.detach().cpu().numpy().astype(np.uint16))
        
