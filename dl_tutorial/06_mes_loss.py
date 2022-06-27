import torch
import torch.nn as nn

loss = nn.MSELoss()
input = torch.ones(2, 2)
target = torch.ones(2, 2) + 0.1
target[0,0] = 1.8
output = loss(input, target)
print(output)