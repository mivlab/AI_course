import torch
import cv2
from torch.autograd import Variable
from torchvision import transforms
from models.cnn import Net

model = Net()
model.load_state_dict(torch.load('output/params_10.pth'))
# model = torch.load('output/model.pth')
model.eval()
if torch.cuda.is_available():
    model.cuda()
img = cv2.imread('4_00440.jpg')
img_tensor = transforms.ToTensor()(img)
img_tensor = img_tensor.unsqueeze(0)
if torch.cuda.is_available():
    prediction = model(Variable(img_tensor.cuda()))
else:
    prediction = model(Variable(img_tensor))
pred = torch.max(prediction, 1)[1]
print(pred)
cv2.imshow("image", img)
cv2.waitKey(0)