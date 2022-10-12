import cv2 as cv
import numpy as np
def face_detect_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")
    faces = face_detector.detectMultiScale(gray, 1.02, 5)
    for x, y, w, h in faces:
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv.imshow("result", image)

capture = cv.VideoCapture(0)
while (True):
    #按帧读取视频，ret,frame是获cap.read()方法的两个返回值。其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
    ret, frame = capture.read()
    # cv.flip函数表示图像翻转，沿y轴翻转, 0: 沿x轴翻转, <0: x、y轴同时翻转
    #frame = cv.flip(frame, 1)
    face_detect_demo(frame)
    #waitKey（）方法本身表示等待键盘输入，参数是1，表示延时1ms切换到下一帧图像，对于视频而言；
    c = cv.waitKey(10)
    if c == 27:#当键盘按下‘ESC’退出程序
        break


