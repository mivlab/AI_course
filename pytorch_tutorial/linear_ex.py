import torch
import torch.nn as nn
from torchvision import transforms
import  numpy_tes as np
import cv2

li = nn.Linear(3, 2)
net = nn.Sequential(li)
li.weight.requires_grad_(False)
li.bias.requires_grad_(False)
li.weight.fill_(0.1)
li.bias.fill_(0.1)
#li.weight[0][0] = 0.5
#li.weight[0][1] = 0.5
#li.bias[0] = 1

data = np.array([20, 20, 10], dtype=np.float32)
data1 = torch.tensor(data)
out = net(data1)
print(out)

#for idx, m in enumerate(net.modules()):
#    print(m)
#for param in net.parameters():
#    print(param)
