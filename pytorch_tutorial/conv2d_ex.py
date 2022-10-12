import torch
import torch.nn as nn
from torchvision import transforms
import  numpy_tes as np
import cv2

conv1 = nn.Conv2d(3, 2, 3)
net = nn.Sequential(conv1)
conv1.weight.requires_grad_(False)
conv1.bias.requires_grad_(False)
conv1.weight.fill_(0.1)
conv1.bias.fill_(0.1)

# RGB数据，格式为chw, 3x4x4
data = np.array([[[200, 200, 10, 10],
                   [200, 200, 10, 10],
                   [200, 200, 10, 10],
                   [200, 200, 10, 10]],
                  [[10, 10, 200, 200],
                   [20, 20, 200, 200],
                   [10, 10, 200, 200],
                   [20, 20, 200, 200]],
                  [[10, 10, 10, 10],
                   [20, 20, 10, 10],
                   [10, 10, 10, 10],
                   [20, 20, 10, 10]]],
                 dtype=np.float32)

image = torch.tensor(data)
images = image.unsqueeze(0) # 由chw 变为 nchw， n=1
out = net(images)
print(out)

# 保存图像
data1 = np.swapaxes(data, 0, 2) # 变为whc
data2 = np.swapaxes(data1, 0, 1) # 变为hwc
data2 = cv2.cvtColor(data2, cv2.COLOR_RGB2BGR)
cv2.imwrite('1.jpg', data2)
