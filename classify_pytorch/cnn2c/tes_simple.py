import torch
import cv2
from torch.autograd import Variable
from torchvision import transforms
from train_simple import Net_simple

use_cuda = False
model = Net_simple(10)
# 注意：此处应把pth文件改为你训练出来的params_x.pth，x为epoch编号，
# 一般来讲，编号越大，且训练集（train）和验证集（val）上准确率差别越小的（避免过拟合），效果越好。
model.load_state_dict(torch.load('output/params_simple.pth'))
model.eval()
if use_cuda and torch.cuda.is_available():
    model.cuda()

# 保存网络参数
f = open('1.txt', 'w')
shape = model.conv.weight.shape #权重的维度 16*3*3*3，依次为输出通道16、输入通道3、卷积核3x3
for i in range(shape[0]):
    weight = model.conv.weight[i].reshape(-1) #把权重变成一维数组
    for j in range(shape[1] * shape[2] * shape[3]):
        f.write(f'{weight[j].item()} ') #保存权重
    f.write('\n')
    f.write(f'{model.conv.bias[i].item()}\n') #保存偏置项
f.close()

img = cv2.imread('2.jpg')
img = cv2.resize(img, (28, 28))
img_tensor = transforms.ToTensor()(img)
img_tensor = img_tensor.unsqueeze(0)
if use_cuda and torch.cuda.is_available():
    prediction = model(Variable(img_tensor.cuda()))
else:
    prediction = model(Variable(img_tensor))
pred = torch.max(prediction, 1)[1]
print(prediction)
print(pred)
cv2.imshow("image", img)
cv2.waitKey(0)