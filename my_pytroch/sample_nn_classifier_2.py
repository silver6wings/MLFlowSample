import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.optim as optim

import time


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def imshow(img):
    # 反归一化
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # 归一化 平均值0.5 标准差0.5 后续为什么反归一化为 img = img / 2 + 0.5 原因在这里
        # 这里应该是 batch Normalization 为了使训练更容易 使用归一化
        #                        x - mean
        # Batch Normalization = ----------
        #                          std
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

transet = torchvision.datasets.CIFAR10(root='./data/train', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data/test', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(transet, batch_size=4, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 展示一些图片
# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

for image, label in trainloader:
    imshow(image[0])

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# # 第一步：创建网络
# net = Net()
# # net.to(device)
# # 第二部：创建损失函数 这里用的交叉熵
# criterion = nn.CrossEntropyLoss()
# # criterion.to(device)
# # 第三步：创建权重更新规则 这里用的SGD
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# # 第四步：循环迭代数据 在数据迭代器上循环传给网络和优化器输入
# since = time.time()
# for epoch in range(2):
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         # inputs = inputs.to(device)
#         # labels = labels.to(device)
#
#         # 核心代码
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         if i % 2000 == 1999:
#             print('[{0},{1}] loss:{2}'.format(epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
# time_elapsed = time.time() - since
# print('Training complete in {:.0f}m {:.0f}s'.format(
#     time_elapsed // 60, time_elapsed % 60))
# print('Finished Training')
#
