'''
习题：生成一个向量，对它做ReLU，演示其运行结果。
要求对比手工计算和程序实现的结果。
'''

import torch
import torch.nn as nn
m = nn.ReLU()
input = torch.randn(2)
output = m(input)
print(input)
print(output)