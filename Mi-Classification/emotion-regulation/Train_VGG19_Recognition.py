'''
按照之前已经训练好的模型参数可知
最好的超参数有两种：
1、'best_PublicTest_acc_epoch': 3, 'best_PrivateTest_acc_epoch': 4, 'best_PublicTest_acc': tensor(99.9162)
2、'best_PublicTest_acc_epoch': 21, 'best_PrivateTest_acc_epoch': 77, 'best_PublicTest_acc': tensor(99.9444)
'''

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import VGG
from Variables import *
from datasets_preprocessing import DataAndLabel
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import matplotlib.pyplot as plt

cTime = time.time()
train = DataAndLabel(data_path='train_datas.csv', label_path='train_labels_onehot.csv')
val = DataAndLabel(data_path='val_datas.csv', label_path='val_labels_onehot.csv')
print(len(train), len(val))
train_load = DataLoader(train, batch_size=batch_size, shuffle=False)
val_load = DataLoader(val, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = VGG("VGG19")
loss_func = CrossEntropyLoss()  # 交叉熵损失函数
optimizer = Adam(params=net.parameters(), lr=lr)  # 优化器
net = net.to(device)

for i in range(train_epochs):
    print(f"==============第{i+1}轮训练===============")
    train_loss = 0
    net.train()
    for Input, Target in train_load:
        Input = Input.to(device)
        Target = Target.to(device)
        Output = net(Input)
        # 优化
        optimizer.zero_grad()
        loss = loss_func(Output, Target)
        loss.backward()
        optimizer.step()

        # 记录损失
        train_loss += loss.item()
    loss_history.append(train_loss / len(train))
    print("当前批次数据训练损失：", loss_history[i])

    net.eval()
    total_accuracy = 0
    with torch.no_grad():
        for Input, Target in val_load:
            Input_v = Input.to(device)
            Target_v = Target.to(device)
            Output = net(Input_v)

            accuracy = (Output.argmax(1) == Target_v.argmax(1)).sum()
            total_accuracy += accuracy

    acc.append(total_accuracy / len(val))
    print(f'当前批次验证精度：', acc[i])

plt.plot(range(train_epochs), torch.tensor(loss_history, device='cpu'))
plt.title('Train Loss')
plt.show()

plt.plot(range(train_epochs), torch.tensor(acc, device='cpu'))
plt.ylim([0., 1.])
plt.title('Validation Accuracy')
plt.show()

dTime = time.time()
Time = dTime - cTime
print(Time)

torch.save(net.state_dict(), 'PrivateTest_weight.pth')  # 保存模型参数
