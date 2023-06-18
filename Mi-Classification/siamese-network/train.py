import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from SiameseCBAM import *

print('包导入完成！')

# 数据导入
train_dataset = SiameseNetworkDataset(file_path='A01_train_new100.pt',
                                      target_path='A01_train22_one_hot_label.pt',
                                      transform=transforms.ToTensor())
train_dataloader = DataLoader(train_dataset, shuffle=False, num_workers=0, batch_size=128)

eval_dataset = SiameseNetworkDataset(file_path='A01_test_new100.pt',
                                     target_path='A01_test22_one_hot_label.pt',
                                     transform=transforms.ToTensor())
eval_dataloader = DataLoader(eval_dataset, shuffle=False, num_workers=0, batch_size=128)

device = ('cuda' if torch.cuda.is_available() else 'cpu')
net = SiameseNetwork().to(device)
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

train_loss_history = []  # 记录训练损失
train_accuracy_history = []  # 记录训练集分类精度
eval_loss_history = []  # 记录验证集损失
eval_accuracy_history = []  # 记录验证集分类精度

for epoch in range(0, 200):
    train_loss = 0
    train_accuracy = 0
    # 训练
    net.train()
    for i, train_data in enumerate(train_dataloader, 0):
        input1, input2, train_label = train_data
        input1, input2, train_label = input1.to(device), input2.to(device), train_label.to(device)
        output1, output2 = net(input1, input2)  # 特征提取

        optimizer.zero_grad()
        loss = criterion(output1, output2, train_label)  # 每一批训练样本损失值
        loss.backward()
        optimizer.step()  # 优化

        pred_1, pred_2 = output1.argmax(1), output2.argmax(1)  # 预测是否相似
        pred = (pred_1 != pred_2)  # 相似为 0 ，不相似为 1
        accuracy = (pred == train_label).sum()  # 每一批训练样本分类精度
        train_accuracy += accuracy.item()  # 训练样本每一轮分类精度

        train_loss += loss.item()  # 训练样本每一轮损失
    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy / 288)
    if epoch % 10 == 0:
        print("训练周期{} \n本轮训练总损失{} ".format(epoch + 1, train_loss))
        print("本轮训练精度{} ".format(train_accuracy / 288))

    eval_loss = 0
    eval_accuracy = 0
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(eval_dataloader, 0):
            item1, item2, label = data
            item1, item2, label = item1.to(device), item2.to(device), label.to(device)

            output1, output2 = net(item1, item2)
            predict1, predict2 = output1.argmax(1), output2.argmax(1)
            pred_label = (predict1 != predict2)  # 不像似为 1
            accuracy = (pred_label == label).sum()
            eval_accuracy += accuracy.item()

            loss = criterion(output1, output2, label)
            eval_loss += loss.item()
        eval_loss_history.append(eval_loss)
        eval_accuracy_history.append(eval_accuracy / 288)

        if epoch % 10 == 0:
            print("本轮测试损失{} ".format(eval_loss))
            print('本轮验证精度{} '.format(eval_accuracy / 288))

# 绘图
fig = plt.figure(figsize=(8, 8), dpi=100)
ax_1 = fig.add_subplot(221)
plt.plot(range(len(train_loss_history)), train_loss_history)
plt.title('Train Loss')
ax_2 = fig.add_subplot(222)
plt.plot(range(len(train_accuracy_history)), train_accuracy_history)
plt.title('Train Accuracy')
ax_3 = fig.add_subplot(223)
plt.plot(range(len(eval_loss_history)), eval_loss_history)
plt.title('Evaluate Loss')
ax_4 = fig.add_subplot(224)
plt.plot(range(len(eval_accuracy_history)), eval_accuracy_history)
plt.title('Evaluate Accuracy')
plt.xlabel('epoch')
plt.show()

# 模型保存
torch.save(net.state_dict(), 'The_Siamese_Model_1.pth')
print('模型训练完成!')
