import torch
import torch.nn as nn
m = nn.ReLU()
input = torch.randn(2)
output = m(input)
print(input)
print(output)