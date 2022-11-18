'''
习题：打开一张纯色的彩色图，对它做3x3卷积，卷积系数全为0.1，偏置项也为0.1。
要求对比手工计算和程序实现的结果。
'''

import torch
import torch.nn as nn
import numpy as np
import cv2
img = cv2.imread('2.jpg', cv2.IMREAD_COLOR) #读图像，读出来格式为ndarray
img = img.transpose((2, 0, 1)) # 把HWC转换为CHW，即把第2维提前面，第0、1维放后面
conv = nn.Conv2d(3, 1, 3) #卷积参数：输入3通道，输出1通道，3x3卷积
conv.weight.requires_grad_(False) #不需要计算梯度
conv.bias.requires_grad_(False)
conv.weight[0, :, :, :] = 0.1 #设置所有权重
conv.bias[:] = 0.1 #设置所有偏置
#卷积要求的输入数据格式是NCHW，tensor浮点类型，因此需要转浮点、转tensor、由CHW变为NCHW
out = conv(torch.tensor(img.astype(np.float32)).unsqueeze(0)) #转换为tensor，并增加一维，做卷积，得到结果
print(out)
print(26*0.1*9+132*0.1*9+239*0.1*9+0.1)
