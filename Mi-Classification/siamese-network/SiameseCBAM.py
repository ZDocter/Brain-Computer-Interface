import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from torch import nn


# Custom Contrastive Loss
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        # 这里的labei不是样本标签，是样本相似度标签为  0相似 / 1不相似
        # 相似时，欧式距离小才能使损失小，不同时，欧式距离大才能使损失小（欧式距离小于阈值）
        label = label.view(-1)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class SiameseNetworkDataset(Dataset):
    def __init__(self, file_path, target_path, transform=None, target_transform=None):
        super(SiameseNetworkDataset, self).__init__()
        self.file_path = file_path
        self.target_path = target_path

        self.data = self.parse_data_file(file_path)
        self.target = self.parse_target_file(target_path)

        self.transform = transform
        self.target_transform = target_transform

    def parse_data_file(self, file_path):
        data = torch.load(file_path)  # 导入训练数据
        return np.array(data, dtype=np.float32)  # 返回为32位浮点型

    def parse_target_file(self, target_path):
        target = torch.load(target_path)  # 导入标签数据
        return np.array(target, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item0 = self.data[idx]  # 基准样本
        target = self.target[idx]  # 基准样本标签

        flag = np.random.randint(0, 1)
        # 保证正负样本对各占 50 %
        if flag == 0:
            idx1 = int(np.random.choice(np.arange(len(self.target))[(self.target == target).reshape(-1)], 1))  # 生成正样本对
        else:
            idx1 = int(np.random.choice(np.arange(len(self.target))[(self.target != target).reshape(-1)], 1))  # 生成负样本对

        item1 = self.data[idx1]
        target1 = self.target[idx1]
        label = np.array(0 if target1 == target else 1)  # 相似记为0，不相似记为1

        if self.transform:
            item0 = self.transform(item0).permute(1, 2, 0)  # [100, 10, 22] -- [22, 100, 10]
            item1 = self.transform(item1).permute(1, 2, 0)
        if self.target_transform:
            label = self.target_transform(label)

        return item0, item1, torch.from_numpy(label)


# 获取待测样本
class SiameseNetworkDatasetTest(Dataset):
    def __init__(self, file_path, target_path,
                 a, b,
                 transform=None, target_transform=None):
        super(SiameseNetworkDatasetTest, self).__init__()
        self.file_path = file_path
        self.target_path = target_path
        self.a = a
        self.b = b

        self.data = self.parse_data_file(file_path)
        self.target = self.parse_target_file(target_path)
        self.adata = self.parse_data_file(a)
        self.atarget = self.parse_target_file(b)

        self.transform = transform
        self.target_transform = target_transform

    def parse_data_file(self, file_path):
        data = torch.load(file_path)  # 导入训练数据
        return np.array(data, dtype=np.float32)  # 返回为32位浮点型

    def parse_target_file(self, target_path):
        target = torch.load(target_path)  # 导入标签数据
        return np.array(target, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]  # 基准样本
        target = self.target[idx]  # 基准样本标签

        if self.transform:
            item = self.transform(item).permute(1, 2, 0)  # [100, 10, 22] -- [22, 100, 10]

        if self.target_transform:
            target = self.target_transform(target)

        return item, target


class channel_attention(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接的通道下降倍数
    def __init__(self, in_channel, ratio=4):
        # 继承父类初始化方法
        super(channel_attention, self).__init__()

        # 全局最大池化 [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 第一个全连接层, 通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # 第二个全连接层, 恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        # relu激活函数
        self.relu = nn.ReLU()
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 获取输入特征图的shape
        b, c, h, w = inputs.shape

        # 输入图像做全局最大池化 [b,c,h,w]==>[b,c,1,1]
        max_pool = self.max_pool(inputs)
        # 输入图像的全局平均池化 [b,c,h,w]==>[b,c,1,1]
        avg_pool = self.avg_pool(inputs)

        # 调整池化结果的维度 [b,c,1,1]==>[b,c]
        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])

        # 第一个全连接层下降通道数 [b,c]==>[b,c//4]
        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        # 激活函数
        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        # 第二个全连接层恢复通道数 [b,c//4]==>[b,c]
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        # 将这两种池化结果相加 [b,c]==>[b,c]
        x = x_maxpool + x_avgpool
        # sigmoid函数权值归一化
        x = self.sigmoid(x)
        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])
        # 输入特征图和通道权重相乘 [b,c,h,w]
        outputs = inputs * x

        return outputs


# （2）空间注意力机制
class spatial_attention(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, kernel_size=7):
        # 继承父类初始化方法
        super(spatial_attention, self).__init__()

        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)  # 输入通道数2，输出通道数1，卷积核7*7，图像填充为3（上下左右各3）
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)  # 最大池化，dim=1，按列输出

        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)  # 平均池化
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)  # 横向拼接

        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)  # 卷积运算
        # 空间权重归一化
        x = self.sigmoid(x)
        # 输入特征图和空间权重相乘
        outputs = inputs * x

        return outputs


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.input_channel = 22
        self.ca = channel_attention(self.input_channel)
        self.sa = spatial_attention()

        # 构造自己的网络
        self.cnn1 = nn.Sequential(
            # nn.ReflectionPad2d(1),
            nn.Conv2d(22, 16, kernel_size=2, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(16),  # 批量归一化

            nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),  # 数据归一化

            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64))

        # 在全连接层之前再次加入CBAN机制
        self.ca1 = channel_attention(self.input_channel)
        self.sa1 = spatial_attention()
        # 构造全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(128, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(100, 100),
            nn.Sigmoid(),

            nn.Linear(100, 4))

    def forward(self, Input1, Input2):
        output1 = self.cnn1(Input1)
        output1 = output1.view(output1.size()[0], -1)
        Output1 = self.fc1(output1)

        output2 = self.cnn1(Input2)
        output2 = output2.view(output2.size()[0], -1)
        Output2 = self.fc1(output2)
        return Output1, Output2
