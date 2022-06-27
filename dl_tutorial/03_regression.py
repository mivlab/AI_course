
import torch
import math
import torch.nn as nn

dense = nn.Linear(2, 1)
#dense.weight.requires_grad_(False)
#dense.bias.requires_grad_(False)

x = torch.Tensor([0.6, 0.8])

y = dense(x)

print(y)
