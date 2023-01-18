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


def expand(img):
    h = img.shape[0]
    w = img.shape[1]
    img1 = np.zeros((h,w),np.uint8)
    etch_b = np.array([[0,1,0],[1,1,1],[0,1,0]])
    for i in range(h-2):
        for j in range(w-2):
            a = (img[i:i+3,j:j+3] * etch_b).sum()
            if a==255:
                for c in range(3):
                    for d in range(3):
                        if etch_b[c,d] == 1:
                            img1[i+c,j+d] = etch_b[c,d]*255
    return img1


def etch(img):
    h = img.shape[0]
    w = img.shape[1]
    img1 = np.zeros((h,w),np.uint8)
    etch_b = np.array([[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]])
    for i in range(h-4):
        for j in range(w-4):
            a = (img[i:i+5,j:j+5] * etch_b).sum()
            if a==9*255:
                img1[i+2,j+2] = 255
            else:
                img1[i+2,j+2] = 0
    return img1


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data_in       = sample['image_in']
        name          = sample['image_name']
        # data_gt = sample['groundtruth']
        # return {'image_in': torch.from_numpy(data_in),'groundtruth':data_gt,'image_name':name}
        return {'image_in': torch.from_numpy(data_in),'image_name':name}

class ReconsDataset(torch.utils.data.Dataset):
     def __init__(self, all_data_path, transform, in_size):
        self.all_data_path = all_data_path
        self.transform = transform
        self.in_size = in_size
        self.dirs_data = os.listdir(self.all_data_path)
        
     def __len__(self):
         dirs = self.dirs_data
         return len(dirs)          

     def __getitem__(self, idx):
         dirs = self.dirs_data 
         file_name = os.path.join(self.all_data_path, dirs[idx]+'/light.tif')
         data_in = io.imread(file_name)
         max_intensity = max(data_in.flatten())
         data_in = data_in/max_intensity
         
         sample = {'image_in': data_in, 'image_name':dirs[idx]}
         
         if self.transform:
             sample = self.transform(sample)
         return sample


if __name__ == "__main__":
    cuda = torch.device('cuda:0')
    batch_size = 1
    input_size= 1
    output_size = 2
    SRRFDATASET = ReconsDataset(all_data_path="/home/jlh/DBSN/Cropped_images_Nucl_Memb/Testing/",
                                transform = ToTensor(),
                                in_size = 256)   
   
    test_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size, shuffle=False, pin_memory=True) # better than for loop
    model = UNet(n_channels=input_size, n_classes=output_size)
    
# for thre_1 in range(5,250,5):
    thre_1 = 45
    thre_2 = thre_1
    # model_num = 5000
    # for thre_2 in range (5,250,5):
    for model_num in range(10,5001,10):
        print(model_num)
        model.cuda(cuda)
        model.load_state_dict(torch.load("/home/jlh/DBSN/Models/Nucleus_Membrane_255/Models_"+str(model_num)+".pkl"))
        model.eval()
        file_path = '/home/jlh/DBSN/Prediction/ALL_Nucleus_Membrane_255_'+str(thre_1)+'-'+str(thre_2)+'/Pred_'+str(model_num)+'/'
        # file_path = '/home/jlh/20220916/Prediction/NM_255/Nucleus_Membrane_255_'+str(thre_1)+'-'+str(thre_2)+'/'
        folder = os.path.exists(file_path)
        if not folder:
            os.makedirs(file_path)
        
        for batch_idx, items in enumerate(test_dataloader):
            image        = items['image_in']
            image_name   = items['image_name']
            
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
            
            I_1 = pred[0,0,:,:]
            # print(min(I_1.flatten()))
            # print(max(I_1.flatten()))
            I_1[I_1>=thre_1] = 255
            I_1[I_1<thre_1]  = 0
            
            I_2 = pred[0,1,:,:]
            # print(min(I_2.flatten()))
            # print(max(I_2.flatten()))
            I_2[I_2>=thre_2] = 255
            I_2[I_2<thre_2]  = 0
            I_2 = I_2 - I_1
            I_2[I_2<0] = 0
            
            # I_1 = expand(I_1)
            # I_1 = etch(I_1)
            image_name = prediction_save_path+'/Nucleus.tif'
            io.imsave(image_name, I_1.detach().cpu().numpy().astype(np.uint8))
            
            # I_2 = expand(I_2)
            # I_2 = etch(I_2)
            image_name = prediction_save_path+'/Membrane.tif'
            io.imsave(image_name, I_2.detach().cpu().numpy().astype(np.uint8))
               
 
       
        
