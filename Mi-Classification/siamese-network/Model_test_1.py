import numpy as np
import torch.cuda
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from sklearn import manifold
from SiameseCBAM import *

siamese_dataset = SiameseNetworkDataset(file_path='A01_test_new100.pt',
                                        target_path='A01_test22_one_hot_label.pt',
                                        transform=transforms.ToTensor())
test_dataloader = DataLoader(siamese_dataset, shuffle=False, num_workers=0, batch_size=128)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = SiameseNetwork()
net.load_state_dict(torch.load('The_Siamese_Model.pth'))

net = net.to(device)
criterion = ContrastiveLoss()

test_loss_history = []
test_accuracy_history = []

for i in range(50):
    test_loss = 0
    test_accuracy = 0
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            item1, item2, label = data
            item1, item2, label = item1.to(device), item2.to(device), label.to(device)

            output1, output2 = net(item1, item2)
            predict1, predict2 = output1.argmax(1), output2.argmax(1)
            pred_label = (predict1 != predict2)  # 不像似为 1
            accuracy = (pred_label == label).sum()
            test_accuracy += accuracy.item()

            loss = criterion(output1, output2, label)
            test_loss += loss.item()
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy / 288)
        print("本轮测试总损失 ", test_loss)

acc_average = 0
for i in test_accuracy_history:
    acc_average += i
print('平均分类精度 ', acc_average / 50)

fig = plt.figure(figsize=(8, 8), dpi=100)
# ax_1 = fig.add_subplot(221)
# plt.plot(range(len(test_loss_history)), test_loss_history)
# plt.ylim((0, 20))
# ax_2 = fig.add_subplot(222)
plt.plot(range(len(test_accuracy_history)), test_accuracy_history)
plt.ylim((0, 1))
plt.title("Average test accuracy: {:.2%}".format(acc_average / 50))
plt.show()
