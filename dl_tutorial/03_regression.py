
import torch
import math
import torch.nn as nn

dense = nn.Linear(2, 1)
dense.weight.requires_grad_(False)
dense.bias.requires_grad_(False)
dense.weight[0, 0] = 0.1
dense.weight[0, 1] = 0.2
dense.bias[0] = 0.1
x = torch.Tensor([0.6, 0.8])
y = dense(x)
print(0.6 * 0.1 + 0.8 * 0.2 + 0.1)
print(y)
