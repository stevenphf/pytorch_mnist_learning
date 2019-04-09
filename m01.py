#-*-coding:utf-8-*-

import time
print(time.ctime())
t=time.time()

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision

USE_CUDA = torch.cuda.is_available()
# USE_CUDA = False

torch.manual_seed(1)

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist/', #保存位置
    train=True, #training set
    transform=torchvision.transforms.ToTensor(), #converts a PIL.Image or numpy.ndarray
                                                 #to torch.FloatTensor(C*H*W) in range(0.0,1.0)
    download=DOWNLOAD_MNIST
)

# print(train_data.data.size())
# print(train_data.targets.size())
# plt.imshow(train_data.data[10].numpy(),cmap='gray')
# plt.title('%i' % train_data.targets[10])
# plt.show()

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist/',train=False)
test_x = Variable(torch.unsqueeze(test_data.data, dim=1)).type(torch.FloatTensor)[:2000]/255.   #volatile=True
test_y = test_data.targets[:2000]

if USE_CUDA: test_x, test_y = test_x.cuda(), test_y.cuda()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1,28,28)
            nn.Conv2d(in_channels=1,  # input height
                      out_channels=16,  # n_filter
                      kernel_size=5,  # filter size
                      stride=1,  # filter step
                      padding=2  # con2d出来的图片大小不变； if stride = 1, pad = (ker_size-1)/2 = (5-1)/2, 保证输出图和输入的一致
                      ),  # output shape (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 2x2采样，output shape (16,14,14)
        )

        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32,14,14)
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))   # output shape (32,7,7)

        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)    # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)  # flat (batch_size, 32*7*7)
        output = self.out(x)
        return output

if USE_CUDA:
    cnn=CNN().cuda()
else:
    cnn = CNN()
# print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

#for (b_x,b_y) in train_loader:

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        if USE_CUDA:
            b_x, b_y = Variable(x).cuda(), Variable(y).cuda()
        else:
            b_x, b_y = Variable(x), Variable(y)  #50, 1, 28, 28

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            # accuracy = sum(pred_y == test_y)/test_y.size(0)/1.0
            accuracy = (pred_y == test_y).sum().item()/1.0/test_y.size(0)    # 默认都是int类型
            # print('Epoch: ', epoch, ' | train loss: %.4f '% loss.item(), ' | accuracy: %.4f'% accuracy)
            print('Epoch: {} | train loss: {:.3f} | accuracy: {:.2%}'.format(epoch, loss.item(),accuracy))

test_output = cnn(test_x[:10])
if USE_CUDA:
    pred_y = torch.max(test_output.cpu(), 1)[1].data.squeeze()
else:
    pred_y = torch.max(test_output,1)[1].data.squeeze()

print(pred_y,'prediction number')

if USE_CUDA:
    print(test_y[:10].cpu().numpy(),'real number')
else:
    print(test_y[:10].numpy(),'real number')

print(time.ctime())
print(time.time()-t)