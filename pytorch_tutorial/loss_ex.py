
import torch
import torch.nn as nn
import numpy_tes as np

data = torch.tensor([1.0, -1.0, -1.0]).unsqueeze(0)
y = torch.tensor([2])
w=torch.Tensor([1.0, 2.0, 3.0])
loss_func = nn.CrossEntropyLoss(weight=w)
loss = loss_func(data, y)
print(loss)







#import math
#print(-math.log(math.exp(-1) / (math.exp(1) + math.exp(-1) + math.exp(-1))))