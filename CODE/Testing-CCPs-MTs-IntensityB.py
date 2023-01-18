#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        I_1           = sample['image_CCPs']
        I_2           = sample['image_MTs']
        name          = sample['image_name']
        max_intensity = sample['max_intensity']
        
        return {'image_in': torch.from_numpy(data_in),
                'image_CCPs': torch.from_numpy(I_1),
                'image_MTs': torch.from_numpy(I_2),
                'image_name':name,
                'max_intensity':max_intensity}

class ReconsDataset(torch.utils.data.Dataset):
     def __init__(self, all_data_path, transform, in_size):
        self.all_data_path = all_data_path
        self.transform = transform
        self.in_size = in_size
        self.dirs_data = os.listdir(self.all_data_path+'Tesing_GT_CCPs/')
     def __len__(self):
         dirs = self.dirs_data
         return len(dirs)          

     def __getitem__(self, idx):
         dirs = self.dirs_data 
         
         file_name = os.path.join(self.all_data_path+'Tesing_GT_CCPs/', dirs[idx])
         I_1 = io.imread(file_name)
         file_name = os.path.join(self.all_data_path+'Tesing_GT_MTs/', dirs[idx])
         I_2 = io.imread(file_name)
         
         ratio_1 = np.random.randint(9)/10+0.1
         ratio_2 = np.random.randint(9)/10+0.1
         I_1 = I_1*ratio_1
         I_2 = I_2*ratio_2
         data_in = I_1+I_2
         
         max_intensity = max(data_in.flatten())
         data_in = data_in/max_intensity
         
         sample = {'image_in': data_in, 'image_CCPs': I_1, 'image_MTs': I_2, 'image_name':dirs[idx][:-4],'max_intensity':max_intensity}
         
         if self.transform:
             sample = self.transform(sample)
         return sample


if __name__ == "__main__":
    cuda = torch.device('cuda:0')
    batch_size = 1
    input_size= 1
    output_size = 1
    SRRFDATASET = ReconsDataset(all_data_path="/home/jlh/DBSN/Cropped_images/",
                                transform = ToTensor(),
                                in_size = 256)   
   
    test_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size, shuffle=False, pin_memory=True) # better than for loop
    model = UNet(n_channels=input_size, n_classes=output_size)
    
    # print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    for model_num in range(0,5001,10):
        print(model_num)
        model.cuda(cuda)
        model.load_state_dict(torch.load("/home/jlh/DBSN/Models/CCPs_MTs-DL-intensity/Models_"+str(model_num)+".pkl"))
        model.eval()
        file_path = '/home/jlh/DBSN/Prediction/CCPs_MTs-DL-intensity/Pred_'+str(model_num)+'/'
        folder = os.path.exists(file_path)
        if not folder:
            os.makedirs(file_path)
        
        for batch_idx, items in enumerate(test_dataloader):
            image         = items['image_in']
            image_1       = items['image_CCPs']
            image_2       = items['image_MTs']
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
            image_name = prediction_save_path+'/Output.tif'
            io.imsave(image_name, I.detach().cpu().numpy().astype(np.uint16))
            image_name = prediction_save_path+'/CCPs.tif'
            io.imsave(image_name, image_1.detach().cpu().numpy().astype(np.uint16))
            image_name = prediction_save_path+'/MTs.tif'
            io.imsave(image_name, image_2.detach().cpu().numpy().astype(np.uint16))
            
        
