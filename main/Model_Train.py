import warnings
warnings.simplefilter('ignore')
import pandas as pd
import torch
import numpy as np
import torchvision
import os
import torch.nn as nn
import math
from tqdm import tqdm
import torchvision.models as models
import matplotlib.pyplot as plt
import PIL.Image as Image
import skimage.io as io
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import cv2
from models.ResNet_CAG import *
from utils.aux_func import *
from utils.dataset import IFEDataset



def train(args):
    device = torch.device(args["device"])

    if not os.path.isdir(args["model_save_dir"]):
        os.makedirs(args["model_save_dir"])

    total_label = []
    total_predictions = []

    file = pd.read_csv(args["img_info_path"])
    Y_data = file["MULTI-RESULT"].values
    fold = file["FOLD"].values

    if args["diagonal_input"]:
        X_data = np.load(args["data_path"])
        X_data = X_data * get_mask(X_data.shape[-1])
    else:
        X_data = np.load(args["data_path"])

    for i_fold in range(10):
        model_save_path =  '{}ResNet_fold_{}.pkl'.format(args["model_save_dir"], i_fold)
        torch.cuda.empty_cache()
        train_index = fold != i_fold
        val_index = fold == i_fold

        print('********************************************************************')
        print('****************************{} fold start***************************'.format(i_fold))
        print('********************************************************************')

        best_val_loss = 100.
        best_val_score = 0.
        best_epoch_pred = []

        train_loss_list = []
        val_loss_list = []
        train_acc_list = []

        val_acc_list = []
        train_f1_list = []
        val_f1_list = []

        # prepare train data and val data
        x_train = X_data[train_index]  # train data
        x_val = X_data[val_index]
        y_train = Y_data[train_index]  # label
        y_val = Y_data[val_index] 

        # dataset defination
        train_dataset = IFEDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
        val_dataset = IFEDataset(x_val, y_val)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)

        # model
        model = ResidualNet(network_type='ImageNet', depth=18, num_classes=9, att_type=args["attn_type"], use_mask=args["use_cag"])
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        if args["optim"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args["lr"])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args["mile_stones"], gamma=0.1)

        epoch_label = np.array(y_val).tolist()
        for epoch in range(args["total_epoch"]):
            #每次迭代的评价指标
            running_loss = 0.
            val_running_loss = 0.
            epoch_pred = []
            train_label = []
            train_pred = []

            print("*************************************************************")
            print("{} epoch train start:".format(epoch))
            print(scheduler.get_lr())

            # ************** start to train ********************
            model.train()

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                outputs = model(x)
                _, pred = torch.max(outputs, 1)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.shape[0]
                train_label.extend(y.data.cpu().numpy().tolist())
                train_pred.extend(pred.data.cpu().numpy().tolist())

            scheduler.step()
            # update evaluation criteria
            epoch_loss = running_loss / len(y_train)
            train_acc = accuracy_score(train_label, train_pred)
            train_f1 = f1_score(train_label, train_pred, average='macro')
            # update list
            train_loss_list.append(epoch_loss)
            train_f1_list.append(train_f1)
            train_acc_list.append(train_acc)
            # print
            print('train loss : {:.4f}, train acc : {:.4f}, train f1 : {:.4f}'.format(epoch_loss, train_acc, train_f1))

            # ************** start to validate *****************
            model.eval()
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)   
                prob = model(x)
                _, pred = torch.max(prob, 1)
                val_loss = criterion(prob, y)
                val_running_loss += val_loss.item() * x.shape[0]
                epoch_pred.extend(pred.data.cpu().numpy().tolist())
            # update evalution criteria
            val_epoch_loss = val_running_loss / len(y_val)
            epoch_acc = accuracy_score(epoch_label, epoch_pred)
            epoch_f1 = f1_score(epoch_label, epoch_pred, average='macro')
            # update list
            val_loss_list.append(val_epoch_loss)
            val_acc_list.append(epoch_acc)
            val_f1_list.append(epoch_f1)
            # print
            print('val loss : {:.4f}, val acc : {:.4f}, val f1 : {:.4f}'.format(val_epoch_loss, epoch_acc, epoch_f1))

            # *************** save the best model ************************ 
            if epoch_f1 > best_val_score:
                best_val_score = epoch_f1
                print('best score now !!!!!!!!!!!!!!************!!!!!!!!!'.format(epoch))
                torch.save(model.state_dict(), model_save_path)
                best_epoch_pred = epoch_pred

        # draw loss and print metrics every fold 
        print(classification_report(epoch_label, best_epoch_pred, digits=4))
        draw_loss(args["total_epoch"], train_loss_list, val_loss_list, True)    
        draw_acc(args["total_epoch"], train_acc_list, val_acc_list, 'acc')
        draw_acc(args["total_epoch"], train_f1_list, val_f1_list, 'f1')

        # update total fold matrix
        total_label.extend(epoch_label)
        total_predictions.extend(best_epoch_pred)
    # print total fold metrics
    print(classification_report(total_label, total_predictions, digits=4))