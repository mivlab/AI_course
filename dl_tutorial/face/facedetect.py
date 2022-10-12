import cv2
import numpy as np
from train_faceexpres import Net48
import torch
from torchvision import transforms

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    model = Net48()
    model.load_state_dict(torch.load('params_5.pth'))
    model.eval()
    expres = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    while True:
        flag, frame = cap.read()  # 检测读取
        if not flag:
            break
        gray = cv2.cvtColor(frame, code=cv2.COLOR_BGRA2GRAY)
        gray3 = cv2.cvtColor(gray, code=cv2.COLOR_GRAY2BGR)
        #  转化为灰度进行检测 ，检测效果可调下方参数
        face = face_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8)
        #data = face_detector.detectMultiScale(frame)

        for x, y, w, h in face:
            cv2.rectangle(frame, pt1=(x, y),
                          pt2=(x + w, y + h),
                          color=[0, 0, 255],  # 框选的颜色RGB可自调，不会可以百度搜索
                          thickness=2)  # 线框的粗度
            fa = cv2.resize(gray3[y:y + h, x:x + w, :], (48, 48))
            img_tensor = transforms.ToTensor()(fa)
            img_tensor = img_tensor.unsqueeze(0)
            prediction = model(img_tensor)
            pred = torch.max(prediction, 1)[1]
            print(expres[pred])
        #  打印人脸位置坐标
        print(face)
        cv2.imshow('face', frame)
        key = cv2.waitKey(1000 // 24)  # 整除
        #  键盘输入a退出
        if key == ord('a'):
            break
    cv2.destroyAllWindows()
    cap.release()
