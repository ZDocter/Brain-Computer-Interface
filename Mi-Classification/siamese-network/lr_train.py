import matplotlib.pyplot as plt
import torch.cuda
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from SiameseCBAM import *


def Train():
    m = [0.1, 0.5, 1.0, 1.5, 2.0]
    fig = plt.figure(figsize=(8, 8), dpi=100)
    for rate in m:
        print('当前m ', rate)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = SiameseNetwork().to(device)
        criterion = ContrastiveLoss(margin=rate)
        optimizer = optim.Adam(net.parameters(), lr=0.0005)

        loss_history = []
        acc_history = []
        eval_loss_history = []
        total_loss = 0

        net.train()
        for epoch in range(0, 40):
            total_loss = 0
            for i, data in enumerate(train_dataloader, 0):
                item1, item2, label = data
                item1, item2, label = item1.to(device), item2.to(device), label.to(device)

                output1, output2 = net(item1, item2)
                optimizer.zero_grad()
                loss = criterion(output1, output2, label)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()  # Loss of every train epoch
            loss_history.append(total_loss)

        plt.plot(range(len(loss_history)), loss_history, label=rate)

            # eval_loss = 0
            # eval_accuracy = 0
            # net.eval()
            # with torch.no_grad():
            #     for i, data in enumerate(eval_dataloader, 0):
            #         item1, item2, label = data
            #         item1, item2, label = item1.to(device), item2.to(device), label.to(device)
            #
            #         output1, output2 = net(item1, item2)
            #         loss = criterion(output1, output2, label)
            #         eval_loss += loss.item()
            #     eval_loss_history.append(eval_loss)
        # plt.plot(range(len(eval_loss_history)), eval_loss_history, label=rate)
    # plt.title('Loss with different learning_rate')
    plt.title('Loss with different margin')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    siamese_dataset = SiameseNetworkDataset(file_path='A01_train_new100.pt',
                                            target_path='A01_train22_one_hot_label.pt',
                                            transform=transforms.ToTensor())
    train_dataloader = DataLoader(siamese_dataset, shuffle=False, num_workers=0, batch_size=128)
    # siamese_dataset = SiameseNetworkDataset(file_path='A01_test_new100.pt',
    #                                         target_path='A01_test22_one_hot_label.pt',
    #                                         transform=transforms.ToTensor())
    # eval_dataloader = DataLoader(siamese_dataset, shuffle=False, num_workers=0, batch_size=128)
    Train()
