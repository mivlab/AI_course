facedet.py ： opencv的传统方法的人脸检测 CascadeClassifier，非深度学习方法。

facedet_yu.py ： opencv的于仕琪老师的深度学习人脸检测效果展示，效果较好，速度很快。

facedet_yu_mask.py ： opencv的于仕琪老师的深度学习人脸检测，加上我们自己训练的人脸是否戴口罩识别。

用到了face_detection_yunet_2023mar.onnx 深度学习模型，
这个模型的下载方法是，opencv的文档 https://docs.opencv.org/
选择你安装的opencv版本，例如 https://docs.opencv.org/4.8.0/
搜索 FaceDetectorYN，就会看到文档里有介绍如何下载这个模型：
DNN-based face detector.
model download link: https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet