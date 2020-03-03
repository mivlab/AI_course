口罩佩戴检测代码。


使用步骤
1）github上下载 Ultra-Light-Fast-Generic-Face-Detector-1MB代码
https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
2）把run_video_face_detect_mask.py， cnn.py， params_25.pth 三个文件复制到Ultra-Light-Fast-Generic-Face-Detector-1MB代码目录下
3）运行run_video_face_detect_mask.py 即可实现佩戴口罩演示效果。


附：文件介绍
gen_sample.py 根据原图和xml标记文件，抠图保存为固定大小的人脸，存储到face 和face_mask 两个目录下，用于生成二分类训练图像，用到的原图为极市平台口罩识别竞赛的数据。
run_video_face_detect_mask.py 摄像头实时视频口罩佩戴检测演示
cnn.py 二分类网络结构定义