#-*-coding:utf-8-*-
import time
print(time.ctime())
t=time.time()

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torchvision import datasets
from torchvision import transforms
# import matplotlib.pyplot as plt

# 可视化
USE_boardX = True
USE_boardX = False

if USE_boardX:
    # import numpy as np
    # import torchvision.models as models
    # import torchvision.utils as vutils
    from tensorboardX import SummaryWriter  #
    writer = SummaryWriter()   #定义一个SummaryWriter() 实例
    # log_dir为生成的文件所放的目录，comment为文件名称。默认目录为生成runs文件夹目录。

USE_CUDA = torch.cuda.is_available()
# USE_CUDA = False

torch.manual_seed(1)

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_data = datasets.MNIST(
    root='./mnist/', #保存位置
    train=True, #training set
    transform=transforms.ToTensor(), #converts a PIL.Image or numpy.ndarray
                                                 #to torch.FloatTensor(C*H*W) in range(0.0,1.0)
    download=DOWNLOAD_MNIST
)

# print(train_data.data.size())
# print(train_data.targets.size())
# plt.imshow(train_data.data[10].numpy(),cmap='gray')
# plt.title('%i' % train_data.targets[10])
# plt.show()

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = datasets.MNIST(root='./mnist/',train=False)
test_x = Variable(torch.unsqueeze(test_data.data, dim=1)).type(torch.FloatTensor)[:2000]/255.   #volatile=True

test_x = test_x.view(-1,28,28)    # RNN

test_y = test_data.targets[:2000]

if USE_CUDA: test_x, test_y = test_x.cuda(), test_y.cuda()

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=28,
            hidden_size = 64,
            num_layers = 1,
            batch_first = True
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        # r_out = [BATCH_SIZE, input_size, hidden_size]
        # r_out[:, -1, :] = [BATCH_SIZE, hidden_size]  '-1'，表示选取最后一个时间点的 r_out 输出
        out = self.out(r_out[:, -1, :])
        # out = [BATCH_SIZE, 10]
        return out


if USE_CUDA:
    rnn=RNN().cuda()
else:
    rnn = RNN()
# print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):   # 0~1199
        pred = x.view(-1,28,28)
        if USE_CUDA:
            b_x, b_y = Variable(pred).cuda(), Variable(y).cuda()
        else:
            b_x, b_y = Variable(x), Variable(y)  #50, 1, 28, 28

        # 可视化
        if USE_boardX:
            with SummaryWriter(comment='rnn') as w:  # 命名为cnn  ；with 语句，可以避免因w.close未写造成的问题。
                if USE_CUDA:
                    w.add_graph(rnn, (x.cuda(),))
                else:
                    w.add_graph(rnn, (x,))  # 第一个参数为需要保存的模型，第二个参数为输入值，元祖类型。
            for name, param in rnn.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), step)
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()   #将梯度初始化为零
        loss.backward()    #反向传播求梯度
        optimizer.step()   #更新所有参数

        if step % 50 == 0:
            #可视化
            if USE_boardX:
                writer.add_scalar('data/train_loss',loss.item(),step)

            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == test_y).sum().item()/1.0/test_y.size(0)    # 默认都是int类型

            # 可视化
            if USE_boardX:
                writer.add_scalar('data/test_accuracy', accuracy, step)
            print('Epoch: {} | train loss: {:.3f} | accuracy: {:.2%}'.format(epoch, loss.item(),accuracy))

# if USE_boardX:
# writer.add_scalar('data/train_loss',loss.item(),step)
# writer.add_scalar('data/test_accuracy', accuracy, step)

test_output = rnn(test_x[:10])
if USE_CUDA:
    pred_y = torch.max(test_output.cpu(), 1)[1].data.squeeze()
else:
    pred_y = torch.max(test_output,1)[1].data.squeeze()

print(pred_y,'prediction number')

if USE_CUDA:
    print(test_y[:10].cpu().numpy(),'real number')
else:
    print(test_y[:10].numpy(),'real number')

if USE_boardX:
    writer.close()

print(time.ctime())
print(time.time()-t)
