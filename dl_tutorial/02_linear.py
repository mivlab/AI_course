'''
习题：定义一个全连接层，其权重和偏置都是0.1，对比手工计算和pytorch运算的结果
'''

import torch
import torch.nn as nn

input = torch.FloatTensor([1, 2, 3])
li = nn.Linear(3, 2)
li.weight.requires_grad = False
li.bias.requires_grad = False
li.weight[:] = 0.1
li.bias[:] = 0.1
output = li(input)
print(input)
print(output)
