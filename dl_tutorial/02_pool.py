'''
习题：生成一个4x4矩阵，对它做kernel_size为2的最大池化，演示其运行结果。
'''

import torch
import torch.nn as nn
x = torch.arange(1.,17.,1.0).reshape(4, 4)
x = x.unsqueeze(0).unsqueeze(0)
pl = nn.MaxPool2d(2)
y = pl(x)
print(x)
print(y)

