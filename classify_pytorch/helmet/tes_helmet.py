import torch
import cv2
from torch.autograd import Variable
from torchvision import transforms
from train_helmet import Net64x64
from toonnx import to_onnx

use_cuda = False
model = Net64x64()
model.load_state_dict(torch.load('output/params_helmet_21.pth', map_location=torch.device('cpu')))
# model = torch.load('output/model.pth')
model.eval()
if use_cuda and torch.cuda.is_available():
    model.cuda()

to_onnx(model, 3, 64, 64, 'output/params_helmet.onnx')

img = cv2.imread('20220516_2_4_0.jpg') #('7_1.jpg')
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