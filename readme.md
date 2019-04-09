# 文件说明
m01.py是不加可视化的数字识别在pytorch的实现
m02.py是加可视化的数字识别在pytorch的实现
这两个文件都支持cuda的自动识别，速度相差一倍以上

# tensorboardX 运行：
在主目录下，runs目录外运行：tensorboard --logdir runs

# python3 vs python2
python2/3都可以run tensorboardX
python2/3都可以run m01.py和m02.py
python2 速度快于 python3

# 使能调试开关
USE_CUDA    :CUDA开关 
USE_boardX  :可是化开关

# CUDA
CUDA开启后，可视化boardX相关参数也要用.cuda， 如x.cuda()
CUDA下的，需要变：
    小批量数据格式,如：b_x.cuda()
    test数据格式,如：test_y.cuda()
    网络：cnn().cuda()
    以及 可视化 add 的 全部x数据: x.cuda()
    
# Result
Tue Apr  9 16:45:10 2019
Epoch: 0 | train loss: 2.310 | accuracy: 6.05%
Epoch: 0 | train loss: 0.618 | accuracy: 83.25%
Epoch: 0 | train loss: 0.127 | accuracy: 87.35%
Epoch: 0 | train loss: 0.237 | accuracy: 91.35%
Epoch: 0 | train loss: 0.405 | accuracy: 92.80%
Epoch: 0 | train loss: 0.085 | accuracy: 93.80%
Epoch: 0 | train loss: 0.195 | accuracy: 94.55%
Epoch: 0 | train loss: 0.109 | accuracy: 94.80%
Epoch: 0 | train loss: 0.123 | accuracy: 95.95%
Epoch: 0 | train loss: 0.069 | accuracy: 96.35%
Epoch: 0 | train loss: 0.224 | accuracy: 96.20%
Epoch: 0 | train loss: 0.210 | accuracy: 96.45%
Epoch: 0 | train loss: 0.023 | accuracy: 96.65%
Epoch: 0 | train loss: 0.085 | accuracy: 97.05%
Epoch: 0 | train loss: 0.214 | accuracy: 96.95%
Epoch: 0 | train loss: 0.104 | accuracy: 97.00%
Epoch: 0 | train loss: 0.041 | accuracy: 97.20%
Epoch: 0 | train loss: 0.090 | accuracy: 97.85%
Epoch: 0 | train loss: 0.059 | accuracy: 97.75%
Epoch: 0 | train loss: 0.099 | accuracy: 97.60%
Epoch: 0 | train loss: 0.032 | accuracy: 97.80%
Epoch: 0 | train loss: 0.023 | accuracy: 97.70%
Epoch: 0 | train loss: 0.025 | accuracy: 98.30%
Epoch: 0 | train loss: 0.119 | accuracy: 98.15%
(tensor([7, 2, 1, 0, 4, 1, 4, 9, 5, 9]), 'prediction number')
(array([7, 2, 1, 0, 4, 1, 4, 9, 5, 9]), 'real number')
Tue Apr  9 16:45:45 2019
35.6973719597_

# tensorboardX
##Scalars
![scalars](images/scalars.png)

![graphs](images/graphs.png)

![distributions](images/distributions.png)

![histograms](images/histograms.png)

