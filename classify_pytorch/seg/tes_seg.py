import torch
import cv2
from torch.autograd import Variable
from torchvision import transforms
from train_seg import Net_seg
#from toonnx import to_onnx

use_cuda = False
model = Net_seg()
model.load_state_dict(torch.load('output/params_1.pth'))
# model = torch.load('output/model.pth')
model.eval()
if use_cuda and torch.cuda.is_available():
    model.cuda()

img = cv2.imread('../4_00440.jpg')
img = cv2.resize(img, (128, 128))
img_tensor = transforms.ToTensor()(img)
img_tensor = img_tensor.unsqueeze(0)
if use_cuda and torch.cuda.is_available():
    prediction = model(Variable(img_tensor.cuda()))
else:
    prediction = model(Variable(img_tensor))
prediction = prediction.squeeze(3).squeeze(2)
pred = torch.max(prediction, 1)[1]
print(pred)
print(prediction)
cv2.imshow("image", img)
cv2.waitKey(0)