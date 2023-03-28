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
        data_gt       = sample['image_gt']
        name          = sample['image_name']
        max_intensity = sample['max_intensity']
        
        return {'image_in': torch.from_numpy(data_in),
                'image_gt': torch.from_numpy(data_gt),
                'image_name':name,
                'max_intensity':max_intensity}
class ReconsDataset(torch.utils.data.Dataset):
     def __init__(self, all_data_path, transform):
        self.all_data_path = all_data_path
        self.transform = transform
        self.dirs_data = os.listdir(self.all_data_path+'ER/')
        
     def __len__(self):
         dirs = self.dirs_data 
         return len(dirs)          

     def __getitem__(self, idx):
         dirs = self.dirs_data 
         file_name = os.path.join(self.all_data_path+'ER/', dirs[idx])
         I_1 = io.imread(file_name)
         file_name = os.path.join(self.all_data_path+'Adhesion/', dirs[idx])
         I_2 = io.imread(file_name)
         
         max_intensity_1 = max(I_1.flatten())
         max_intensity_2 = max(I_2.flatten())
         max_intensity = max(max_intensity_1,max_intensity_2)
           
         ratio_1 = np.random.randint(9)/10+0.1
         ratio_2 = np.random.randint(9)/10+0.1
         
         data_in = I_1/max_intensity_1*ratio_1+I_2/max_intensity_2*ratio_2
         data_gt = I_1/max_intensity_1+I_2/max_intensity_2
         
         
         data_in = data_in/max(data_in.flatten())
         data_gt = data_gt/max(data_gt.flatten())*max_intensity
         
         sample = {'image_in': data_in, 'image_gt': data_gt, 'image_name':dirs[idx][:-4],'max_intensity':max_intensity}
         
         if self.transform:
             sample = self.transform(sample)
         return sample

if __name__ == "__main__":
    cuda = torch.device('cuda:0')
    batch_size = 1
    input_size= 1
    output_size = 1
    SRRFDATASET = ReconsDataset(all_data_path="/home/jlh/20221208/0-SIM-YongXin/Testing_ER_Adhesion-2/",
                                transform = ToTensor())  
   
    test_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size, shuffle=False, pin_memory=True) # better than for loop
    model = UNet(n_channels=input_size, n_classes=output_size)
    
    # print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    for model_num in range(0,5001,10):
        print(model_num)
        model.cuda(cuda)
        model.load_state_dict(torch.load("/home/jlh/20221208/Models/YXSIM_transfer/IB/Models_"+str(model_num)+".pkl"))
        model.eval()
        file_path = '/home/jlh/20221208/Prediction/YXSIM_transfer/IB/Pred_'+str(model_num)+'/'
        folder = os.path.exists(file_path)
        if not folder:
            os.makedirs(file_path)
        
        for batch_idx, items in enumerate(test_dataloader):
            image         = items['image_in']
            image_gt      = items['image_gt']
            image_name    = items['image_name']
            max_intensity = items['max_intensity']
            
            prediction_save_path = os.path.join(file_path, image_name[0])
            folder = os.path.exists(prediction_save_path)
            if not folder:
                os.makedirs(prediction_save_path)
            model.train()
            image = image.unsqueeze(dim=3)
            image = np.swapaxes(image, 1,3)
            image = np.swapaxes(image, 2,3)
            image = image.float()
            image = image.cuda(cuda)  
            
            pred = model(image)
            
            max_intensity = max_intensity.float()
            max_intensity = max_intensity.cuda(cuda)  
            
            I = pred[0,0,:,:]
            mv = I.flatten().min()
            if mv < 0:
                I = I + abs(mv)
            I = I*max_intensity
            image = image*max_intensity
            image_name = prediction_save_path+'/Output.tif'
            io.imsave(image_name, I.detach().cpu().numpy().astype(np.uint16))
            image_name = prediction_save_path+'/Input.tif'
            io.imsave(image_name, image.detach().cpu().numpy().astype(np.uint16))
            image_name = prediction_save_path+'/GT.tif'
            io.imsave(image_name, image_gt.detach().cpu().numpy().astype(np.uint16))
            
            
        
