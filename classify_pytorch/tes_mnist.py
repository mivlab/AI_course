import torch
import cv2
from torch.autograd import Variable
from torchvision import transforms
from models.cnn import Net
from toonnx import to_onnx

use_cuda = False
model = Net(10)
model.load_state_dict(torch.load('output/params_1.pth'))
# model = torch.load('output/model.pth')
model.eval()
if use_cuda and torch.cuda.is_available():
    model.cuda()

to_onnx(model, 3, 28, 28, 'output/params.onnx')

img = cv2.imread(r'D:\project\AI_course\classify_pytorch\4_00440.jpg')
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