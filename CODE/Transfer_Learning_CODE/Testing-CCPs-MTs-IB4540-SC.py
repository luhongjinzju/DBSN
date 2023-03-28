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
        # return {'image_in': torch.from_numpy(data_in),'groundtruth':data_gt,'image_name':name}
        return {'image_in': torch.from_numpy(data_in),'image_name':name,'max_intensity':max_intensity}

class ReconsDataset(torch.utils.data.Dataset):
     def __init__(self, all_data_path, transform):
        self.all_data_path = all_data_path
        self.transform = transform
        self.dirs_data = os.listdir(self.all_data_path+'CCPs/')
        
     def __len__(self):
         dirs = self.dirs_data 
         return len(dirs)          

     def __getitem__(self, idx):
         dirs = self.dirs_data 
         
         file_name = os.path.join(self.all_data_path+'CCPs/', dirs[idx])
         I_1 = io.imread(file_name)
         file_name = os.path.join(self.all_data_path+'MTs/', dirs[idx])
         I_2 = io.imread(file_name)
         
         max_intensity_1 = max(I_1.flatten())
         max_intensity_2 = max(I_2.flatten())
         max_intensity = max(max_intensity_1,max_intensity_2)
          
         data_in = I_1/max_intensity_1+I_2/max_intensity_2
         data_in = data_in/max(data_in.flatten())
         
         sample = {'image_in': data_in, 'image_name':dirs[idx][:-4],'max_intensity':max_intensity}
         
         if self.transform:
             sample = self.transform(sample)
         return sample
         

if __name__ == "__main__":
    cuda = torch.device('cuda:0')
    batch_size = 1
    input_size= 1
    output_size = 2
    SRRFDATASET = ReconsDataset(all_data_path="/home/jlh/20220916/0-SIM-YongXin/Testing_CCPs_MTs-2/",
                                transform = ToTensor())   
   
    test_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size, shuffle=False, pin_memory=True) # better than for loop
    model = UNet(n_channels=input_size, n_classes=output_size)
    
    model_IB = UNet(n_channels=input_size, n_classes=1)
    model_IB.cuda(cuda)
    model_IB.load_state_dict(torch.load("/home/jlh/20220916/Models/YXSIM_transfer/IB/Models_4540.pkl"))
    model_IB.eval()


    # print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    for model_num in range(10,5001,10):
        print(model_num)
        model.cuda(cuda)
        model.load_state_dict(torch.load("/home/jlh/20220916/Models/YXSIM_transfer/SC-2/Models_"+str(model_num)+".pkl"))
        model.eval()
        file_path = '/home/jlh/20220916/Prediction/SIM_YongXin_testing/YXSIM_transfer/SC-2/Pred_'+str(model_num)+'/'
        folder = os.path.exists(file_path)
        if not folder:
            os.makedirs(file_path)
        
        for batch_idx, items in enumerate(test_dataloader):
            image        = items['image_in']
            image_name   = items['image_name']
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
            image = model_IB(image)
            
            pred = model(image)
            
            max_intensity = max_intensity.float()
            max_intensity = max_intensity.cuda(cuda)  
            
            I = pred[0,0,:,:]
            mv = I.flatten().min()
            if mv < 0:
                I = I + abs(mv)
            I = I*max_intensity
            image_name = prediction_save_path+'/MTs.tif'
            io.imsave(image_name, I.detach().cpu().numpy().astype(np.uint16))
            
            I = pred[0,1,:,:]
            mv = I.flatten().min()
            if mv < 0:
                I = I + abs(mv)
            I = I*max_intensity
            image_name = prediction_save_path+'/CCPs.tif'
            io.imsave(image_name, I.detach().cpu().numpy().astype(np.uint16))
             
            image = image*max_intensity
            image_name = prediction_save_path+'/Input.tif'
            io.imsave(image_name, image.detach().cpu().numpy().astype(np.uint16))
       
        
