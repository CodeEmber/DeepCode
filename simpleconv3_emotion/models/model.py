'''
Author       : wyx-hhhh
Date         : 2023-04-29
LastEditTime : 2024-01-14
Description  : 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

## 3层卷积神经网络simpleconv3定义
## 包括3个卷积层，3个BN层，3个ReLU激活层，3个全连接层


class simpleconv3(nn.Module):
    ## 初始化函数
    def __init__(self, nclass):
        super(simpleconv3, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, 2)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, 3, 2)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 48, 3, 2)
        self.bn3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(48 * 5 * 5, 1200)
        self.fc2 = nn.Linear(1200, 128)
        self.fc3 = nn.Linear(128, nclass)

    ## 前向函数
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 48 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    net = simpleconv3(10)
    y = net(x)
    print(net)
