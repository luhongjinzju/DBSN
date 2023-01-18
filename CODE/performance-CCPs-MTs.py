#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from xlwt import *
import numpy as np
import os
import math
from skimage import io, transform
from skimage.measure import compare_ssim
from PIL import Image

def psnr(img1, img2):
    img1 = (img1/np.amax(img1))*255
    img2 = (img2/np.amax(img2))*255
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def nrmse(img_gt, img2, type="sd"):
    mse = np.mean( (img_gt - img2) ** 2 )
    rmse = math.sqrt(mse)
    if type == "sd":
        nrmse = rmse/np.std(img_gt)
    if type == "mean":
        nrmse = rmse/np.mean(img_gt)
    if type == "maxmin":
        nrmse = rmse/(np.max(img_gt) - np.min(img_gt))
    if type == "iq":
        nrmse = rmse/ (np.quantile(img_gt, 0.75) - np.quantile(img_gt, 0.25))
    if type not in ["mean", "sd", "maxmin", "iq"]:
        print("Wrong type!")
    return nrmse


def ssim(img1, img2, data_range = None):
    #if img2.min() < 0:
    #   img2 += abs(img2.min())
    img2 = (img2/img2.max()) * img1.max()
    #img1 = (img1/img1.max()) * 255
    if data_range is None:
        score = compare_ssim(img1, img2)
    else:
        score = compare_ssim(img1, img2, data_range = data_range)
    return score

if __name__ == "__main__":
    
    gt_path="/media/star/DBSN/DL_Intensity_balance/Data/0-Data_4_training/CCPs-MTsDens/Cropped_images/Testing_Input/"
    dirs = os.listdir(gt_path) 
    p_value_avg = np.zeros((501, 6))
    
    for pred_num in range(10,5001,10):
        pred_path = "/media/star/DBSN/DL_Intensity_balance/Data/0-Data_4_training/CCPs-MTsDens/Prediction_CCPs_MTs-DL-intensity/Pred_"+ str(pred_num) + "/"
        data_save = "/media/star/DBSN/DL_Intensity_balance/Data/0-Data_4_training/CCPs-MTsDens/Performance/Prediction_Intensity/Pred_"+ str(pred_num) + '.xls'
        file = Workbook(encoding = 'utf-8')
        p_value = np.zeros((len(dirs), 3))
        for idx in range(len(dirs)):
            image_name = os.path.join(gt_path, dirs[idx])
            data_gt = io.imread(image_name)
            data_gt = np.array(data_gt)
            image_name = os.path.join(pred_path, dirs[idx][:-4]+'/CCPs.tif')
            data_in = Image.open(image_name)
            data_in = np.array(data_in)
            image_name = os.path.join(pred_path, dirs[idx][:-4]+'/MTs.tif')
            data_in_tmp = Image.open(image_name)
            data_in_tmp = np.array(data_in)
            data_in = data_in + data_in_tmp
            
            min_v = np.quantile(data_gt, 0.01)
            max_v = np.quantile(data_gt, 0.998)
            data_gt = (data_gt - min_v)/(max_v - min_v)
            
            min_v = np.quantile(data_in, 0.01)
            max_v = np.quantile(data_in, 0.998)
            data_in = (data_in - min_v)/(max_v - min_v)
            
            
            p_value[idx,0] = psnr(data_gt, data_in)
            p_value[idx,1] = nrmse(data_gt, data_in)
            p_value[idx,2] = ssim(data_gt, data_in, data_range = data_gt.max())
            
        p_value_avg[int(pred_num/10),0] = p_value[:,0].mean()
        p_value_avg[int(pred_num/10),1] = p_value[:,0].std()            
        p_value_avg[int(pred_num/10),2] = p_value[:,1].mean()
        p_value_avg[int(pred_num/10),3] = p_value[:,1].std()
        p_value_avg[int(pred_num/10),4] = p_value[:,2].mean()
        p_value_avg[int(pred_num/10),5] = p_value[:,2].std()
        table = file.add_sheet('score')
        for k,p in enumerate(p_value):
            for j,q in enumerate(p):
                    table.write(k,j,q)
        file.save(data_save)
    data_save = "/media/star/DBSN/DL_Intensity_balance/Data/0-Data_4_training/CCPs-MTsDens/Performance/Prediction_Intensity/avg_all.xls"
    file = Workbook(encoding = 'utf-8')
    table = file.add_sheet('all')
    for k,p in enumerate(p_value_avg):
        for j,q in enumerate(p):
            table.write(k,j,q)
    file.save(data_save)

   
