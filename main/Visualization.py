import warnings
warnings.simplefilter('ignore')
import pandas as pd
import torch
import numpy as np
import torchvision
import os
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import PIL.Image as Image
import skimage.io as io
import torch.nn.functional as F
import cv2
from models.ResNet_CAG import *
from tqdm import tqdm
import time
from utils.aux_func import *


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)   
            elif "avgpool" in name.lower():
                        x = module(x)
                        x = x.view(x.size(0),-1)
            else:
                x = module(x)
                
        return target_activations, x

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda, cam_type):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        
        self.cam_type = cam_type
        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)
    
    def get_grad_cam_plus(self, feat_map, grad):
        grad = torch.from_numpy(grad)
        feat_map = torch.from_numpy(feat_map)
        alpha_num = grad.pow(2)   # 512*7*7
        alpha_denom = 2 * alpha_num + torch.sum(feat_map.mul(grad.pow(3)))   #512*7*7
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num.div(alpha_denom)
        positive_grad = F.relu(grad)
        weights = (alpha * positive_grad).view(grad.shape[0], -1).sum(dim=1)  # 512
        return weights.numpy()
    
    
    def __call__(self, input, index=None):
        # forward
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        
        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        
        # backward
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        
        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        # get target feature map gradients
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()  #  # 1 * 512 * 7 * 7
        
        target = features[-1]
        target = target.cpu().data.numpy()[0, :] # 512 * 7 * 7       
        
        # get cam
        if self.cam_type == 'grad-cam++':
            weights = self.get_grad_cam_plus(target, grads_val[-1])
        elif self.cam_type == 'grad-cam':
            weights = np.mean(grads_val, axis=(2, 3))[0, :]  # 512
        cam = np.zeros(target.shape[1:], dtype=np.float32) # 7 * 7

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-6)
        
        return cam, index

    
class Grad_Cam_Main:
    def __init__(self, index, data, csv_path, model_path, use_mask=[True]*4,
                 cam_type = "grad-cam", attn_type=None, use_cuda=False):
        
        self.use_cuda = use_cuda
        self.data = data
        self.size = data.shape
        self.model_path = model_path
        self.cam_type = cam_type  #grad-cam or grad-cam++
        self.attn_type = attn_type
        self.use_mask = use_mask
        self.init_order = ['G-A', 'G-M', 'G-K', 'G-L', 'A-M', 'A-K', 'A-L', 'M-K', 'M-L', 'K-L']
        
        file = pd.read_csv(csv_path)
        self.img_path = file.loc[index, "PATH"]
        self.label = file.loc[index, "MULTI-RESULT"]
        
        
    def show_cam(self, mask):
        if np.max(mask) > 1.:
            mask = mask - np.min(mask)
            mask = mask / (np.max(mask) + 1e-6)
        plt.figure()
        plt.imshow(cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB))
    

    def show_cam_on_image(self, sample, pic_mask):
        img = np.concatenate((sample['G_img'], sample['A_img'], sample['M_img'], sample['K_img'], sample['L_img']), axis=1)
        heatmap = cv2.applyColorMap(cv2.resize(np.uint8(255 * pic_mask), (200,200)), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255 #BGR
        mix_img = heatmap + np.float32(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)) / 255
        mix_img = mix_img / np.max(mix_img)
        """
        plt.figure(figsize=(10,5))
        plt.yticks([])
        plt.xticks([])
        plt.imshow(cv2.cvtColor(np.uint8(255 * mix_img), cv2.COLOR_BGR2RGB))
        """
        cv2.imwrite('../imgs/test/cam_on_img.jpg', np.uint8(255 * mix_img))
        print("cam has been saved into ../imgs/test/cam_on_img.jpg")
        return img
    
    
    def get_img(self):
        img_pre = self.img_path.split('.')[0]
        G_path = '../img_blob/' + img_pre + '-G.jpg'
        A_path = '../img_blob/' + img_pre + '-A.jpg'
        M_path = '../img_blob/' + img_pre + '-M.jpg'
        K_path = '../img_blob/' + img_pre + '-K.jpg'
        L_path = '../img_blob/' + img_pre + '-L.jpg'
        
        G_img = Image.open(G_path)
        A_img = Image.open(A_path)
        M_img = Image.open(M_path)
        K_img = Image.open(K_path)
        L_img = Image.open(L_path)
        sample = {'G_img': G_img,'A_img':A_img, 'M_img':M_img, 'K_img':K_img, 'L_img':L_img}
        return sample
    
    def img_resize(self, img_sample):
        for key in ['G_img', 'A_img', 'M_img', 'K_img', 'L_img']:
            img_sample[key] = np.array(torchvision.transforms.functional.resize(img_sample[key], (200, 40)))
        return img_sample
    
    def preprocess_data(self, data):
        process_data = torch.from_numpy(data).type(torch.FloatTensor)
        process_data.unsqueeze_(0)
        input = process_data.requires_grad_(True)
        return input

    def get_fixed_mask(self, mask):
        step = int(200 / mask.shape[-1])
        final_mask = np.zeros([200,200])

        row_sum = np.sum(mask, axis=0)
        column_sum = np.sum(mask, axis=1)
        binding_value = row_sum + column_sum
        binding = cv2.resize(binding_value.reshape([binding_value.shape[0],1]), (1,200))
        max_binding = np.argmax(binding)

        final_mask[max_binding - 8: max_binding + 8, :] = 1.
        return final_mask
    
    
    def __call__(self):
        model = ResidualNet(network_type='ImageNet', depth=18, num_classes=9, att_type=self.attn_type, use_mask=self.use_mask)
        model.load_state_dict(torch.load(self.model_path))
        
        sim_mask_list = []
        
        x_data = self.data.copy()
        x_data = self.preprocess_data(x_data)
        
        for target_module_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            for target_layer_name in ['0', '1']:
                
                #define module
                for name, module in model._modules.items():
                    if name == target_module_name:
                        target_module = module
                
                grad_cam = GradCam(model=model, feature_module=target_module, \
                                   target_layer_names=[target_layer_name], use_cuda=self.use_cuda, cam_type=self.cam_type)

                target_index = None
                sim_mask, pred= grad_cam(x_data, target_index)
                sim_mask_list.append(sim_mask)
        print("true: " + convert_label_2_text(self.label))
        print("pred: " + convert_label_2_text(pred))
        sim_mask_avg = np.array(sim_mask_list).sum(axis=0)
        
        imgs = self.get_img()
        imgs = self.img_resize(imgs)
        
        pic_mask = self.get_fixed_mask(sim_mask_avg)
        
        #self.show_cam(sim_mask_avg)
        self.show_cam_on_image(imgs, pic_mask)
        return pic_mask