# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:19:49 2020

@author: zxl
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import PIL.Image as Image
from sklearn.metrics.pairwise import pairwise_distances
from utils.aux_func import *
from DarknessDensity import *


class G003CollaborationTensor:
    def __init__(self, root_dir, output_size, G003_imgs, strips, sim_type):
        self.root_dir = root_dir
        self.new_h, self.new_w = output_size
        self.G003_imgs = G003_imgs
        self.strips = strips
        self.sim_type = sim_type
        self.data = np.zeros([len(G003_imgs), 10, strips, strips])
    
    def create_collaboration_tensor(self):
        for i, G003_img in enumerate(tqdm(self.G003_imgs)):
            sample = []
            img_pre = G003_img.split('.')[0]
            for lane in ['G', 'A', 'M', 'K', 'L']:
                blob_path = self.root_dir + img_pre + '-' + lane + '.jpg'
                img = Image.open(blob_path)
                img = img.resize((self.new_w, self.new_h))
                img = np.resize(np.array(img), (self.strips, -1))
                sample.append(img / 255.)
            self.data[i] = self.get_similarity(sample)
        return self.data
    
    def get_similarity(self, sample):
        sim = np.zeros([10, self.strips, self.strips])
        index = 0
        for i in range(4):
            first_img = sample[i]
            for j in range(i+1, 5):
                second_img = sample[j]
                sim[index] = pairwise_distances(X=first_img, Y=second_img, metric=self.sim_type)
                index += 1
        return sim
    
def preprocess_data_soft_mask(data, mask):
    X_data = data.copy()
    size = mask.shape[-1]
    for m in range(X_data.shape[0]):
        index = 0
        for i in range(4):
            for j in range(i+1, 5):
                mask_i = mask[m][i].repeat(size).reshape(size, -1)
                mask_j = mask[m][j].repeat(size).reshape(size, -1).T
                sub_mask = np.multiply(mask_i, mask_j)
                X_data[m, index, :, :] = np.multiply(X_data[m, index, :, :], sub_mask)
                index += 1
    return X_data

def create_similarity_dataset(csv_path="../img_detail/Origin_img_features.csv", save_path="../sim_data/euc_100.npy", sim_type="euclidean", corr_lens=100, mask=True):
    """
    根据csv构建数据集，读取分割后的图像，计算correlation，并附加soft_mask
    parmas:
        csv_path: 存储基本信息的csv文件路径
        sim_type: 采用的相似度计算方式
        corr_lens: 最后获得相似度矩阵的长宽
        
    """
    csv_name = csv_path.split("/")[-1].split(".")[0]

    step = int(200/ corr_lens)
    length = corr_lens * step

    Path, label = get_data(csv_path)

    corr_func = G003CollaborationTensor('../img_blob/', (length, 40), Path, corr_lens, sim_type)
    data = corr_func.create_collaboration_tensor()
    
    if mask:
        cal_density = Darkness_Density('../img_blob/', Path, corr_lens)
        img_mask = cal_density.create_density()
    
        img_mask = img_mask.reshape(data.shape[0], 5, corr_lens, -1).mean(axis=3)
        img_mask = img_mask / np.max(img_mask)
        data = preprocess_data_soft_mask(data, img_mask)
    
    np.save(save_path, data)
    print("similarity data has been saved into " + save_path)
    del data