'''
预处理数据，并创建模型对应数据接口
'''

import pandas as pd
import numpy as np
import torch
# from sklearn.preprocessing import OneHotEncoder, Normalizer
from torch.utils.data import Dataset


# data = pd.read_csv('datasets/fer2013.csv')
# emotion = data['emotion']
# pixels = data['pixels']
# usage = data['Usage']
#
# classes = 7
# train_datas, train_labels, val_datas, val_labels, test_datas, test_labels = [], [], [], [], [], []
# '''
# 第一列为情绪标签
# 第二列为每个像素的值
# 第三列为用途
# '''
# for idx in range(0, len(data)):
#     img = list(map(eval, pixels[idx].split(' ')))  # list
#
#     # 数据集划分
#     if usage[idx] == 'Training':
#         train_datas.append(img)  # 样本
#         train_labels.append(emotion[idx])  # 标签
#
#     elif usage[idx] == "PrivateTest":
#         val_datas.append(img)
#         val_labels.append(emotion[idx])
#
#     else:
#         test_datas.append(img)
#         test_labels.append(emotion[idx])
#
# # 标签转化为独热编码
# train_labels_array = np.array(train_labels).reshape(-1, 1)
# train_labels_onehot = OneHotEncoder().fit_transform(train_labels_array).toarray()
#
# test_labels_array = np.array(test_labels).reshape(-1, 1)
# test_labels_onehot = OneHotEncoder().fit_transform(test_labels_array).toarray()
#
# val_labels_array = np.array(val_labels).reshape(-1, 1)
# val_labels_onehot = OneHotEncoder().fit_transform(val_labels_array).toarray()
# # print(train_labels_onehot)
#
# # 标签存为csv文件
# train_labels_df = pd.DataFrame(train_labels_onehot)
# test_labels_df = pd.DataFrame(test_labels_onehot)
# val_labels_df = pd.DataFrame(val_labels_onehot)
#
# train_labels_df.to_csv('train_labels_onehot.csv', header=False, index=False)
# test_labels_df.to_csv('test_labels_onehot.csv', header=False, index=False)
# val_labels_df.to_csv('val_labels_onehot.csv', header=False, index=False)
# print("标签保存完毕！")
# # print(pd.read_csv('val_labels_onehot.csv', header=None, index_col=None))
#
# # 每个样本的保存为csv文件中的一行
# train_datas_array = np.array(train_datas)  # ndarray
# test_datas_array = np.array(test_datas)
# val_datas_array = np.array(val_datas)
# # print(val_datas_array)
#
# # 归一化
# train_datas_morm = Normalizer().fit_transform(train_datas_array)
# test_datas_morm = Normalizer().fit_transform(test_datas_array)
# val_datas_morm = Normalizer().fit_transform(val_datas_array)
#
# train_datas_df = pd.DataFrame(train_datas_morm)
# test_datas_df = pd.DataFrame(test_datas_morm)
# val_datas_df = pd.DataFrame(val_datas_morm)
#
# train_datas_df.to_csv('train_datas.csv', header=False, index=False)
# test_datas_df.to_csv('test_datas.csv', header=False, index=False)
# val_datas_df.to_csv('val_datas.csv', header=False, index=False)
# print('数据保存完毕！')
# # print(pd.read_csv('val_datas.csv', header=None, index_col=None))


class DataAndLabel(Dataset):
    def __init__(self, data_path, label_path):
        super(DataAndLabel, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.datas = self.load_data(data_path)
        self.labels = self.load_label(label_path)

    # 样本转化为输入特征图
    def load_data(self, data_path):
        datas = pd.read_csv(data_path)
        datas_array = np.array(datas, dtype=np.float32).reshape([-1, 1, 48, 48])
        return torch.from_numpy(datas_array)

    def load_label(self, label_path):
        labels = pd.read_csv(label_path)
        labels_array = np.array(labels, dtype=np.float32)
        return torch.from_numpy(labels_array)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        item = self.datas[index]
        target = self.labels[index]
        return item, target

# train = DataAndLabel(data_path='train_datas.csv', label_path='train_labels_onehot.csv')
# print(train[0])
