# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 18:25:33 2020

@author: zxl
"""

import numpy as np
import cv2
import pandas as pd
import os
from tqdm import tqdm


def get_segmented_G003_img(img):
    prefix = img.split('.')[0]
    segmented_imgs = list()
    if (prefix != None):
        segmented_imgs.append(prefix + '-G.jpg')
        segmented_imgs.append(prefix + '-A.jpg')
        segmented_imgs.append(prefix + '-M.jpg')
        segmented_imgs.append(prefix + '-K.jpg')
        segmented_imgs.append(prefix + '-L.jpg')
        #segmented_imgs.append(prefix + '-SP.jpg')
    return segmented_imgs

def get_col_pixel_sum(gray_img):
        col = gray_img.shape[1]
        row = gray_img.shape[0]
        col_pixel_sum = np.array([0] * row)
        for i in range(row):
            _sum = 0
            for j in range(col):
                _sum += gray_img[i][j]
            col_pixel_sum[i] = _sum
        return col_pixel_sum

'''
    对 灰度图像每一行的灰度值和 数组 进行平滑
'''
def smooth_pixel_sum(col_pixel_sum):
    length = col_pixel_sum.shape[0]
    smoothed_pixel_sum = np.array([0] * length)
    smoothed_pixel_sum[0] = col_pixel_sum[0]
    smoothed_pixel_sum[length - 1] = col_pixel_sum[length - 1]
    for i in range(1, length - 1):
        smoothed_pixel_sum[i] = int((col_pixel_sum[i - 1] + 2 * col_pixel_sum[i] + col_pixel_sum[i + 1]) / 4)
    return smoothed_pixel_sum

    
class Darkness_Density:
    def __init__(self, root_dir, G003_imgs, strips):
        self.root_dir = root_dir
        self.G003_imgs = G003_imgs
        self.strips = strips
        self.density = np.zeros([len(G003_imgs), 5, strips])
    
    def create_density(self):
        step = int(200 / self.strips)
        length = self.strips * step
        for i, G003_img in enumerate(tqdm(self.G003_imgs)):
            
            img_pre = G003_img.split('.')[0]
            index = 0
            mask = np.zeros([5, length])
            
            for lane in ['G', 'A', 'M', 'K', 'L']:
                segmented_path = self.root_dir + img_pre + '-' + lane + '.jpg'
                img = cv2.imread(segmented_path)
                img = cv2.resize(img, (40, length))
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #gray_img = np.array(img.convert('L'))
                col_pixel_sum = get_col_pixel_sum(gray_img)
                col_pixel_sum = smooth_pixel_sum(col_pixel_sum)
                mask[index] = -np.log(col_pixel_sum / 10200)   # type 1
                index += 1
                
            mask = mask.reshape(5, self.strips, -1).mean(axis=2)
            self.density[i] = mask  # 4352 * 5 * output_size
        
        if (not os.path.exists('../sim_data/soft_mask/')):
            os.makedirs('../sim_data/soft_mask/')
            
        #np.save('../sim_data/soft_mask/soft_mask_' + str(self.strips) + '.npy', self.density)
        #print("save file in  ../sim_data/soft_mask/soft_mask_" + str(self.strips) + '.npy')    
        return self.density
        