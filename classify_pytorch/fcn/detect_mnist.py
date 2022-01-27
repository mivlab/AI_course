import torch
import cv2
from torch.autograd import Variable
from torchvision import transforms
from train_mnist import Net_fcn
#from toonnx import to_onnx

use_cuda = False
model = Net_fcn()
model.load_state_dict(torch.load('output/params_17.pth'))
# model = torch.load('output/model.pth')
model.eval()
if use_cuda and torch.cuda.is_available():
    model.cuda()

img = cv2.imread('2.jpg')
img_tensor = transforms.ToTensor()(img)
img_tensor = img_tensor.unsqueeze(0)
if use_cuda and torch.cuda.is_available():
    prediction = model(Variable(img_tensor.cuda()))
else:
    prediction = model(Variable(img_tensor))
m = torch.nn.Softmax(dim=1)
prob = m(prediction)
map, pred = torch.max(prob, 1)
map_show = map.squeeze().detach().numpy() # 此句是为了方便在pycharm里可视化
pred_show = pred.squeeze().detach().numpy() # 此句是为了方便在pycharm里可视化
print(pred)
cv2.imshow("image", img)
cv2.waitKey(0)