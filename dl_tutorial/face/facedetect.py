import cv2
import numpy as np

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')  # 检测
    while True:
        flag, frame = cap.read()  # 检测读取
        if not flag:
            break
        gray = cv2.cvtColor(frame, code=cv2.COLOR_BGRA2GRAY)
        #  转化为灰度进行检测 ，检测效果可调下方参数
        face = face_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8)
        data = face_detector.detectMultiScale(frame)

        for x, y, w, h in face:
            cv2.rectangle(frame, pt1=(x, y),
                          pt2=(x + w, y + h),
                          color=[0, 0, 255],  # 框选的颜色RGB可自调，不会可以百度搜索
                          thickness=2)  # 线框的粗度
        #  打印人脸位置坐标
        print(data)
        cv2.imshow('face', frame)
        key = cv2.waitKey(1000 // 24)  # 整除
        #  键盘输入a退出
        if key == ord('a'):
            break
    cv2.destroyAllWindows()
    cap.release()
