import numpy as np
import cv2

cap = cv2.VideoCapture(0)
ret, img = cap.read()
faceDetector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (img.shape[1], img.shape[0]))
sub = cv2.imread('sub.png')
sub = cv2.resize(sub, (60, 60))
back = cv2.imread('11.png')
back = cv2.resize(back, (img.shape[0] * back.shape[1] // back.shape[0], img.shape[0]))
start = 0
while (cap.isOpened()):
    ret, img = cap.read()
    t1 = cv2.getTickCount()
    faces = faceDetector.detect(img)
    print((cv2.getTickCount() - t1) / cv2.getTickFrequency())
    t1 = cv2.getTickCount()
    if faces[1] is None:
        continue
    faces = faces[1].astype(np.int32)
    cv2.rectangle(img, (faces[0, 0], faces[0, 1]), (faces[0, 0] + faces[0, 2], faces[0, 1] + faces[0, 3]), (255, 0, 0))
    cv2.circle(img, (faces[0, 8], faces[0, 9]), 5, (0, 0, 255), -1)
    img[faces[0, 9]: faces[0, 9] + sub.shape[0], 100: 100+sub.shape[1], :] = sub
    print((cv2.getTickCount() - t1) / cv2.getTickFrequency())
    t1 = cv2.getTickCount()
    win = back[:, start:start + img.shape[1], :]
    img = np.where(win == 255, img, win)
    '''
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if win[i, j, 0] == 255:
                pass
            else:
                img[i, j, :] = win[i, j, :]
    '''
    print((cv2.getTickCount() - t1) / cv2.getTickFrequency())
    t1 = cv2.getTickCount()
    start += 10
    if start + img.shape[1] > back.shape[1]:
        start = 0

    #判断潜水艇的中心是否处在蓝色柱子区域
    if win[faces[0, 9] + sub.shape[0]//2, 100+sub.shape[1] // 2, 0] != 255:
        cv2.putText(img, 'crash',  (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        print('撞了')
    print((cv2.getTickCount() - t1) / cv2.getTickFrequency())
    t1 = cv2.getTickCount()
    cv2.imshow('img', img)
    cv2.waitKey(1)
    print((cv2.getTickCount() - t1) / cv2.getTickFrequency())
    print('end')