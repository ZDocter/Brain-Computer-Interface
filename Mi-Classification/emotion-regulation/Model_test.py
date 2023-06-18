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

test = DataAndLabel(data_path='test_datas.csv', label_path='test_labels_onehot.csv')
test_load = DataLoader(test, batch_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = VGG('VGG19')
net.load_state_dict(torch.load('PrivateTest_weight.pth'))

net.eval()
for i in range(50):
    total_accuracy = 0
    with torch.no_grad():
        for Input, Target in test_load:
            Input_v = Input.to(device)
            Target_v = Target.to(device)
            Output = net(Input_v)

            accuracy = (Output.argmax(1) == Target_v.argmax(1)).sum()
            total_accuracy += accuracy

    acc.append(total_accuracy / len(test))
    print(f'当前批次验证精度：', acc[i])
