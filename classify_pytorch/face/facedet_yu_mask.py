import torch
import numpy as np
import cv2
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 32x14x14
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  # 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x7x7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),  # 64x7x7
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x3x3
        )

        self.dense = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),  # fc4 64*3*3 -> 128
            nn.ReLU(),
            nn.Linear(128, 2)  # fc5 128->10
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)  # 64x3x3
        res = conv3_out.view(conv3_out.size(0), -1)  # batch x (64*3*3)
        out = self.dense(res)
        return out

model = Net()
model.load_state_dict(torch.load('params_16.pth'))
model.eval()

cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret, img = cap.read()
    img1 = cv2.resize(img, (320, 320))
    t1 = cv2.getTickCount()
    faceDetector = cv2.FaceDetectorYN.create("yunet_yunet_final_320_320_simplify.onnx", "", img1.shape[:2])
    faces = faceDetector.detect(img1)
    if faces[1] is None:
        continue
    faces = faces[1].astype(np.int32)[0]
    img2 = cv2.resize(img1[faces[0]:faces[0] + faces[2], faces[1]:faces[1] + faces[3], :], (28, 28))
    cv2.rectangle(img1, (faces[0], faces[1]), (faces[0] + faces[2], faces[1] + faces[3]),
                  (255, 0, 0))
    for i in range(5):
        cv2.circle(img1, (faces[4 + i * 2], faces[4 + i * 2 + 1]), 2, (0, 255, 0))

    img_tensor = transforms.ToTensor()(img2)
    img_tensor = img_tensor.unsqueeze(0)
    prediction = model(Variable(img_tensor))
    pred = torch.max(prediction, 1)[1].item()
    if pred == 0:
        print('未戴口罩')
        cv2.putText(img1, 'face', (faces[0], faces[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), thickness = 2)
    else:
        print('戴口罩')
        cv2.putText(img1, 'mask', (faces[0], faces[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255),thickness = 2)

    cv2.imshow('img1', img1)
    cv2.waitKey(1)
    #print(faces)
    #print((cv2.getTickCount() - t1) / cv2.getTickFrequency())