#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import torch
from torch.utils.data import  DataLoader
from skimage import io

from unet_model import UNet

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data_in, data_out = sample['image_in'], sample['groundtruth']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        #landmarks = landmarks.transpose((2, 0, 1))
        
        #return {'image': image, 'landmarks': torch.from_numpy(landmarks)}
        return {'image_in': torch.from_numpy(data_in),
               'groundtruth': torch.from_numpy(data_out)}

class ReconsDataset(torch.utils.data.Dataset):
     def __init__(self, all_data_path, transform, in_size):
        self.all_data_path = all_data_path
        self.transform = transform
        self.in_size = in_size
        self.dirs_data = os.listdir(self.all_data_path+'GT_CCPs/')
        
     def __len__(self):
         dirs = self.dirs_data 
         return len(dirs)          

     def __getitem__(self, idx):
         dirs = self.dirs_data 
         file_name = os.path.join(self.all_data_path+'Input/', dirs[idx])
         data_gt = io.imread(file_name)
         max_intensity = max(data_gt.flatten())
         data_gt = data_gt/max_intensity
         
         # data_in = np.zeros((self.in_size, self.in_size))
         file_name = os.path.join(self.all_data_path+'GT_CCPs/', dirs[idx])
         I_1 = io.imread(file_name)
         file_name = os.path.join(self.all_data_path+'GT_MTs/', dirs[idx])
         I_2 = io.imread(file_name)
         
         ratio_1 = np.random.randint(9)/10+0.1
         ratio_2 = np.random.randint(9)/10+0.1
         data_in = I_1*ratio_1+I_2*ratio_2
         
         max_intensity = max(data_in.flatten())
         data_in = data_in/max_intensity
         
         sample = {'image_in': data_in, 'groundtruth': data_gt}
         
         if self.transform:
             sample = self.transform(sample)
         return sample

def get_learning_rate(epoch):
    limits = [3, 8, 12]
    lrs = [1, 0.1, 0.05, 0.005]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
        return lrs[-1] * learning_rate

if __name__ == "__main__":
    cuda = torch.device('cuda:0')
    learning_rate = 0.001
    # momentum = 0.99
    # weight_decay = 0.0001
    batch_size = 1
    input_size = 1
    output_size = 1
    SRRFDATASET = ReconsDataset(all_data_path="/home/jlh/DBSN/Cropped_images/",
                                transform = ToTensor(),
                                in_size = 256)
    
    model = UNet(n_channels=input_size, n_classes=output_size)

    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,  betas=(0.9, 0.999))
    for epoch in range(5001):
        train_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size, shuffle=True, pin_memory=False)
        lr = get_learning_rate(epoch)
        for p in optimizer.param_groups:
            p['lr'] = lr
            print("learning rate = {}".format(p['lr']))
            
        for batch_idx, items in enumerate(train_dataloader):
            image = items['image_in']
            gt = items['groundtruth']
            
            model.train()
            image = image.unsqueeze(dim=3)
            image = np.swapaxes(image, 1,3)
            image = np.swapaxes(image, 2,3)
            image = image.float()
            image = image.cuda(cuda)    
            
            gt = gt.unsqueeze(dim=3)
            gt = np.swapaxes(gt, 1,3)
            gt = np.swapaxes(gt, 2,3)
            gt = gt.float()
            gt = gt.cuda(cuda)
            
            pred = model(image).squeeze()

            loss = (pred-gt).abs().mean() + 5 * ((pred-gt)**2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print ("[Epoch %d] [Batch %d/%d] [loss: %f]" % (epoch, batch_idx, len(train_dataloader), loss.item()))
        torch.save(model.state_dict(),"/home/jlh/DBSN/Models/CCPs_MTs-IB/Models_tmp.pkl")
        if epoch in range(0,5001,10):
            torch.save(model.state_dict(),"/home/jlh/DBSN/Models/CCPs_MTs-IB/Models_"+str(epoch)+".pkl")
