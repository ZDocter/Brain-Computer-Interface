import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


# 绘制损失函数变化曲线
def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.title('Contrastive Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # plt.grid('on')
    plt.show()


# 创建数据接口
class Dataset_interface(Dataset):
    def __init__(self, file_path, target_path):
        super(Dataset_interface, self).__init__()
        self.data = self.parse_data_file(file_path)  # 徐训练样本
        self.target = self.parse_target_file(target_path)  # 对应标签

    def parse_data_file(self, file_path):
        data = torch.load(file_path)
        data = np.array(data , dtype=np.float32)
        return torch.from_numpy(data)

    def parse_target_file(self, target_path):
        target = torch.load(target_path)
        target = np.array(target, dtype=np.float32)
        return torch.from_numpy(target)

    def __len__(self):
        return len(self.data)  # 训练集样本量

    def __getitem__(self, index):
        item1 = self.data[index]
        index2 = np.random.choice(len(self.data))  # 随机抽取训练样本中的数据生成item2
        item2 = self.data[index2]
        target = self.target[index]

        return item1, item2, target


# 通道注意力
class channel_attention(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接的通道下降倍数
    def __init__(self, in_channel, ratio=4):
        # 继承父类初始化方法
        super(channel_attention, self).__init__()  # python 2.x

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
        max_pool = max_pool.view(b, c)
        avg_pool = avg_pool.view(b, c)

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
        x = x.view(b, c, 1, 1)
        # 输入特征图和通道权重相乘 [b,c,h,w]
        outputs = inputs * x

        return outputs


# 空间注意力
class spatial_attention(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, kernel_size=7):
        # 继承父类初始化方法
        super(spatial_attention, self).__init__()

        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
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


# 主模型
class SiameseCBAM_network(nn.Module):
    def __init__(self):
        super(SiameseCBAM_network, self).__init__()
        self.inputs_channel = 22
        # 卷积-池化层

        # CBAM
        self.ca1 = channel_attention(self.inputs_channel)
        self.sa1 = spatial_attention()

        self.cnn = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(22, 16, kernel_size=2, stride=2, padding=2)),  # 64*22*100*10--64*16*52*7
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=2)),  # 64*16*52*7--64*16*26*3
            ('batchnorm1', nn.BatchNorm2d(16)),

            ('conv2', nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=2)),  # 64*16*26*3--64*32*15*3
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool2d(kernel_size=2)),  # 64*32*15*3--64*32*7*1
            ('batchnorm2', nn.BatchNorm2d(32)),

            ('conv3', nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=2)),  # 64*32*7*1--64*64*5*2
            ('relu3', nn.ReLU(inplace=True)),
            ('maxpool3', nn.MaxPool2d(kernel_size=2)),  # 64*64*5*2--64*64*2*1
            ('batchnorm3', nn.BatchNorm2d(64))
        ]))

        # CBAM
        self.ca1 = channel_attention(self.inputs_channel)
        self.sa1 = spatial_attention()

        # 全连接层
        self.fc = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(128, 100)),  # 64*128--64*100
            ('relu3', nn.ReLU(inplace=True)),
            ('dropout', nn.Dropout(p=0.5)),

            ('linear2', nn.Linear(100, 100)),  # 64*100--64*100
            ('sigmoid', nn.Sigmoid()),

            ('linear3', nn.Linear(100, 4))  # 64*100--64*4
        ]))

    def forward(self, input1, input2):
        # 输入1
        output1 = self.cnn(input1)  # 64*22*100*10--64*64*2*1
        output1 = output1.view(output1.size()[0], -1)  # 64*128
        output1 = self.fc(output1)  # 64*4
        # 输入2
        output2 = self.cnn(input2)
        output2 = output2.view(output2.size()[0], -1)
        output2 = self.fc(output2)

        return output1, output2


# 对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        euclidean_distance = F.pairwise_distance(output1, output2).reshape(-1, 1)
        loss_contrastive = torch.mean((1 - target) * torch.pow(euclidean_distance, 2) +
                                      target * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        # 限制最小值为0
        return loss_contrastive


# 验证模型
# if __name__ == '__main__':
#     train_model = SiameseCBAM()  # 实例模型
#     optimizer = torch.optim.Adam(train_model.parameters(), lr=0.005)  # 优化器
#     Contrastive_Loss = ContrastiveLoss()  # 损失函数
#     Input1 = torch.ones([2, 22, 100, 10])  # 输入
#     Input2 = torch.zeros([2, 22, 100, 10])
#     target = torch.Tensor(np.array([[1, 0, 0, 0],
#                                     [0, 0, 0, 1]], dtype=np.float32))  # 标签
#
#     Output1, Output2 = train_model(Input1, Input2)
#     optimizer.zero_grad()  # 梯度清零
#     loss = Contrastive_Loss(Output1, Output2, target)
#     loss.backward()  # 误差反向传播
#     optimizer.step()  # 优化
#
#     print(f'Input:{Input1.shape}\ntarget:{target.shape}\noutput:{Output1.shape}\nloss:{loss}')
#
#     # 梯度是否反向传播
#     for name, prams in train_model.fc.named_parameters():
#         print('--name:', name)
#         print('--grad_requires_grad:', prams.requires_grad)
#         print('--grad_valve:', prams.grad)  # 最后一个linear层偏置梯度消失
#
#     for name, prams in train_model.cnn.named_parameters():
#         print('--name:', name)
#         print('--grad_requires_grad:', prams.requires_grad)
#         print('--grad_valve:', prams.grad)

    # 梯度数值非常小
