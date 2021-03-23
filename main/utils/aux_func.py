import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import skimage.io as io
import cv2
import math


def draw_loss(total_epoch, train_loss_list, val_loss_list, dynamic_y=False):
    """
    Args:
        total_epoch (int): iter number of training
        train_loss_list (list/np.array): train loss
        val_loss_list (list/np.array): val loss
        dynamic_y (bool): change the y_max according to the loss 
    return:
        None
    """
    plt.figure()
    if dynamic_y:
        y_max = max(max(val_loss_list), max(train_loss_list))
    else:
        y_max = 2
    x = np.linspace(0, total_epoch, num=total_epoch)
    plt.plot(x, train_loss_list, color='r', label='train_loss')
    plt.plot(x, val_loss_list, 'b', label='val_loss')#'b'指：color='blue'
    plt.legend()  #显示上面的label
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.axis([0, total_epoch, 0, y_max])#设置坐标范围axis([xmin,xmax,ymin,ymax])
    plt.show()


def draw_acc(total_epoch, train_acc, val_acc, ylabel):
    """
    Args:
        total_epoch (int): iter number of training
        train_acc (list/np.array): train accuracy (no matter what kind accuracy)
        val_acc (list/np.array): val accuracy
        ylabel (string): the y label text
    return:
        None
    """
    plt.figure()
    x = np.linspace(0, total_epoch, num=total_epoch)
    plt.plot(x, train_acc, color='r', label='train_'+ylabel)
    plt.plot(x, val_acc, 'b', label='val_'+ylabel)#'b'指：color='blue'
    plt.legend()  #显示上面的label
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.axis([0, total_epoch, 0.7, 1.0])#设置坐标范围axis([xmin,xmax,ymin,ymax])
    plt.show()

def get_mask_pro(size, pro=9):
    """
    Description:
        Generate a mask which follows normal distribution 
    Args:
        size (int): height and width of the target mask 
    """
    mask = np.zeros([size, size])

    u = 0  # 均值μ
    sig = math.sqrt(1)  # 标准差δ
    x = np.linspace(0, u + np.sqrt(pro) * sig, size)
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
    for i in range(size):
        for j in range(size):
            mask[i, j] = y_sig[np.abs(i - j)]
    
    mask = mask / np.max(mask)
    return mask
    
    
def get_mask(size):
    """
    Description:
        Generate a mask which follows normal distribution 
    Args:
        size (int): height and width of the target mask 
    """
    mask = np.zeros([size, size])

    u = 0  # 均值μ
    sig = math.sqrt(1)  # 标准差δ
    x = np.linspace(0, u + 3 * sig, size)
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
    for i in range(size):
        for j in range(size):
            mask[i, j] = y_sig[np.abs(i - j)]
    
    mask = mask / np.max(mask)
    return mask

def convert_label_2_text(label):
    """
    Args:
        label (int): numeric of label
    return: 
        the name of label
    """
    if label == 0:
        return 'Non-M'
    elif label == 1:
        return 'IGG-KAP'
    elif label == 2:
        return 'IGG-LAM'
    elif label == 3:
        return 'IGA-KAP'
    elif label == 4:
        return 'IGA-LAM'
    elif label == 5:
        return 'IGM-KAP'
    elif label == 6:
        return 'IGM-LAM'
    elif label == 7:
        return 'KAP'
    elif label == 8:
        return 'LAM'

def get_band_num(band):
    """
    Args:
        band (string): name of a band like "G"
    return:
        the numeric type of the band. (int)
    
    """
    if band == 'G' or band == 'IGG':
        return 0
    elif band == 'A' or band == 'IGA':
        return 1
    elif band == 'M' or band == 'IGM':
        return 2
    elif band == 'K' or band == 'KAP':
        return 3
    elif band == 'L' or band == 'LAM':
        return 4
    else:
        print("there is no band named {}".format(band))

def get_data(origin_path):
    """
    Args:
        origin_path (string)：path of csv file
    return:
        contribute 'PATH' and 'MULTI-RESULT'
    """
    g003_origin_feature = pd.read_csv(origin_path)
    
    X_index, Y_index = convert_df_2_data(g003_origin_feature)
    return X_index, Y_index

def convert_df_2_data(dataframe):
    """
    Args:
        dataframe (pandas.Dataframe)：dataframe
    return:
        contribute 'PATH' and 'MULTI-RESULT'
    """
    Path = dataframe['PATH'].values
    Y_data = dataframe['MULTI-RESULT'].values
    return Path, Y_data