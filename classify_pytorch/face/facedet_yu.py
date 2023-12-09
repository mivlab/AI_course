import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret, img = cap.read()
    img1 = img #cv2.resize(img, (320, 320))
    faceDetector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (img1.shape[1], img1.shape[0]))
    faces = faceDetector.detect(img1)
    if faces[1] is None:
        continue
    faces = faces[1].astype(np.int32)
    cv2.rectangle(img1, (faces[0, 0], faces[0, 1]), (faces[0, 0] + faces[0, 2], faces[0, 1] + faces[0, 3]),
                  (255, 0, 0))
    cv2.imshow('img', img1)
    cv2.waitKey(1)
    print(faces)