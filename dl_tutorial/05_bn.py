

import torch
import math
import torch.nn as nn

m = nn.BatchNorm2d(2, eps=0, affine=False)

input = torch.randn(2, 2, 2, 2)
output = m(input)
print(input)
print(output)



import numpy as np

print(np.std(np.array([1,2,3]), ddof=1))

e11 = torch.Tensor.mean(input[0,0, :,:])
v11 = torch.Tensor.var(input[0,0, :,:],False)
v11 = math.sqrt(v11.item())

c1 = input[0,0,:,:].flatten().numpy()
c2 = input[0,1,:,:].flatten().numpy()
e1 = np.mean(c1)
v1 = np.std(c1, ddof=1)
e2 = np.mean(c2)
v2 = np.std(c2, ddof=1)
y = (input[0,0,0,0] - e1) / (v11 + m.eps)
print(y)