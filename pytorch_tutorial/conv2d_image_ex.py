import torch
import torch.nn as nn
from torchvision import transforms
import  numpy_tes as np
import cv2

conv1 = nn.Conv2d(3, 2, 3)
net = nn.Sequential(conv1)
conv1.weight.requires_grad_(False)
conv1.bias.requires_grad_(False)
conv1.weight.fill_(1/27.0)
conv1.bias.fill_(0)
p1 = conv1.weight[1,:,:,:]

edge_filter = np.array([[[-1.0, -1.0, -1.0], [-1.0, 9.0, -1.0], [-1.0, -1.0, -1.0]],
            [[-1.0, -1.0, -1.0], [-1.0, 9.0, -1.0], [-1.0, -1.0, -1.0]],
            [[-1.0, -1.0, -1.0], [-1.0, 9.0, -1.0], [-1.0, -1.0, -1.0]]])
edge_filter = edge_filter / 3;
conv1.weight[1,:,:,:] = torch.tensor(edge_filter)

# RGB数据，格式为chw
img = cv2.imread('apple.bmp')
data1 = img.astype(dtype=np.float32)
data2 = np.swapaxes(data1, 0, 2)
data3 = np.swapaxes(data2, 1, 2)
image = torch.tensor(data3)
images = image.unsqueeze(0) # 由chw 变为 nchw， n=1
out = net(images)

# 保存图像
out1 = out.numpy().squeeze(0)
out2 = np.swapaxes(out1, 0, 2) # 变为whc
out3 = np.swapaxes(out2, 0, 1) # 变为hwc
out4 = np.abs(out3).astype(dtype=np.uint8)
cv2.imwrite('2.jpg', out4[:,:,0])
cv2.imwrite('3.jpg', out4[:,:,1])
