import numpy as np
import cv2
sub_vid = cv2.VideoCapture('xjp.mp4')
vid = cv2.VideoCapture(0)
width = 220
height = 300
while (vid.isOpened()):
    rt, sub_frame = sub_vid.read()
    rt2, frame = vid.read()
    if sub_frame is None or frame is None:
        break
    sub_frame = cv2.resize(sub_frame, (width,height))
    frame[:height,:width, :] = sub_frame
    cv2.imshow('true', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
sub_vid.release()
vid.release()

