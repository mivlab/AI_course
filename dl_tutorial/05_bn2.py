import torch

import torch.nn as nn

m = nn.BatchNorm2d(2,affine=True) #权重w和偏重将被使用
input = torch.randn(1,2,3,4)
output = m(input)

print("输入图片：")
print(input)
print("归一化权重：")
print(m.weight)
print("归一化的偏重：")
print(m.bias)

print("归一化的输出：")
print(output)
print("输出的尺度：")
print(output.size())

# i = torch.randn(1,1,2)
print("输入的第一个维度：")
print(input[0][0])
firstDimenMean = torch.Tensor.mean(input[0][0])
firstDimenVar= torch.Tensor.var(input[0][0],False) #Bessel's Correction贝塞尔校正不会被使用

print(m.eps)
print("输入的第一个维度平均值：")
print(firstDimenMean)
print("输入的第一个维度方差：")
print(firstDimenVar)

bacthnormone = \
  ((input[0][0][0][0] - firstDimenMean)/(torch.pow(firstDimenVar+m.eps,0.5) ))\
        * m.weight[0] + m.bias[0]
print(bacthnormone)