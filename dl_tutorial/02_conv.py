import cv2
import numpy as np
import torch
import torch.nn as nn
img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
image = torch.tensor(img.astype(np.float32))
images = image.unsqueeze(0).unsqueeze(0) # 由chw 变为 nchw， n=1
conv = nn.Conv2d(1, 1, 3)
conv.weight.requires_grad_(False)
conv.bias.requires_grad_(False)
conv.weight[0, 0, :, :] = 1.0/ 9
conv.bias[:] = 0
out = conv(images)
out = out.numpy().astype(np.uint8)
cv2.imshow('after', out.squeeze())
cv2.imshow('before', img)
cv2.waitKey()