import cv2
from skimage.measure import label, regionprops
import os
import numpy as np
import pandas as pd
import tqdm
from utils.dtw import *

class DTWImageSegmentation:

    def __init__(self, from_path, to_path):
        '''
        from_path: '../update_img/'
        to_path: '../img_blob/'
        '''
        self.from_path = from_path
        self.to_path = to_path
        
    def get_fixed_sequence(self, length):
        W_sequence = np.zeros([length])
        W_sequence[:] = 5000 
        W_sequence[10: 35] = 17000
        W_sequence[60: 85] = 11500
        W_sequence[115: 140] = 8000
        W_sequence[165: 185] = 6800
        W_sequence[220: 245] = 10500
        W_sequence[270: 300] = 9000

        return W_sequence
    
    
    def segment_resized_G003_img(self, csv_path):
        '''
        分割传入的路径对应的 已经缩放的 G003 image
        :param G003_imgs: list[year + '/' + month + '/' + day + '/' + _type + sample_number + '01.jpg']
        :return:void
        '''
        img_info = pd.read_csv(csv_path)
        G003_imgs = img_info['PATH'].values
        manhattan_distance = lambda x, y: np.abs(x - y)
        cut_point = [47, 104, 156, 208, 260]
        
        j = 0
        for G003_img in tqdm.tqdm(G003_imgs):
            img = cv2.imread(self.from_path + G003_img)
            width = img.shape[1]
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            Q_s = np.sum(255 - gray_img, axis=0)
            W_s = self.get_fixed_sequence(width)
            d, cost_matrix, acc_cost_matrix, min_path = dtw(Q_s, W_s, dist=manhattan_distance)

            points = [0]
            mean_interval = int(width / 6)
            for i in range(5):
                point = int(np.mean(min_path[0][min_path[1] == cut_point[i]]))

                if point - cut_point[i] > 10:
                    point = cut_point[i] + 10
                elif point - cut_point[i] < -10:
                    point = cut_point[i] - 10

                if point - points[-1] > mean_interval + 5:
                    point = points[-1] + mean_interval + 5
                elif point - points[-1] < mean_interval - 5:
                    point = points[-1] + mean_interval - 5
                points.append(point)
            
            # 分割图像
            SP = img[:, points[0] : points[1]]
            G = img[:, points[1] : points[2]]
            A = img[:, points[2] : points[3]]
            M = img[:, points[3] : points[4]]
            K = img[:, points[4] : points[5]]
            L = img[:, points[5] : ]
            
            img_dir = G003_img[0:G003_img.rindex('/')]
            if (not os.path.exists(self.to_path + img_dir)):
                os.makedirs(self.to_path + img_dir)
            img_pre = G003_img.split('.')[0]
            # 保存图像
            SP_path = self.to_path + '{}-SP.jpg'.format(img_pre)  # 图片的写入路径
            cv2.imwrite(SP_path, SP)
            G_path = self.to_path + '{}-G.jpg'.format(img_pre)  # 图片的写入路径
            cv2.imwrite(G_path, G)
            A_path = self.to_path + '{}-A.jpg'.format(img_pre)  # 图片的写入路径
            cv2.imwrite(A_path, A)
            M_path = self.to_path + '{}-M.jpg'.format(img_pre)  # 图片的写入路径
            cv2.imwrite(M_path, M)
            K_path = self.to_path + '{}-K.jpg'.format(img_pre)  # 图片的写入路径
            cv2.imwrite(K_path, K)
            L_path = self.to_path + '{}-L.jpg'.format(img_pre)  # 图片的写入路径
            cv2.imwrite(L_path, L)
            j += 1
        print('共处理%d张图片' % (j))
