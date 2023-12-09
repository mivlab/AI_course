import numpy as np
import cv2

cap = cv2.VideoCapture(0)
ret, img = cap.read()
sub = cv2.imread('sub.png')
sub = cv2.resize(sub, (60, 60))
back = cv2.imread('11.png')
back = cv2.resize(back, (img.shape[1] * back.shape[1]// back.shape[0] , img.shape[1]))
start = 0
while (cap.isOpened()):
    ret, img = cap.read()
    img1 = img #cv2.resize(img, (320, 320))
    t1 = cv2.getTickCount()
    faceDetector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (img1.shape[1], img1.shape[0]))
    faces = faceDetector.detect(img1)
    if faces[1] is None:
        continue
    faces = faces[1].astype(np.int32)
    cv2.rectangle(img1, (faces[0, 0], faces[0, 1]), (faces[0, 0] + faces[0, 2], faces[0, 1] + faces[0, 3]),
                  (255, 0, 0))
    cv2.circle(img1,(faces[0,8],faces[0,9]),10,(0,255,0),-1)   #画实心圆
    img1[faces[0,9]:faces[0,9]+sub.shape[0],100:100+sub.shape[1],:] = sub
    win = back[:, start : start+img1.shape[1], :]
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if win[i, j, 0] == 255 and win[i, j, 1] == 255:
                pass
            else:
                img1[i, j, :] = win[i, j, :]
    start += 60
    if start > back.shape[1] - img1.shape[1]:
        start = 0
    cv2.imshow('img', img1)
    cv2.waitKey(1)
    print(faces)
    print((cv2.getTickCount() - t1) / cv2.getTickFrequency())