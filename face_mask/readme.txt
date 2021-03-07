口罩佩戴检测代码。


使用步骤
1）github上下载 Ultra-Light-Fast-Generic-Face-Detector-1MB代码
https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
2）把run_video_face_detect_mask.py， cnn.py， params_25.pth 三个文件复制到Ultra-Light-Fast-Generic-Face-Detector-1MB代码目录下
3）运行Ultra-Light-Fast-Generic-Face-Detector-1MB下的run_video_face_detect_mask.py 即可实现佩戴口罩演示效果，显示笔记本电脑实时画面。


附：文件介绍
gen_sample.py 根据原图和xml标记文件，抠图保存为固定大小的人脸，存储到face 和face_mask 两个目录下，用于生成二分类训练图像，用到的原图为极市平台口罩识别竞赛的数据。
run_video_face_detect_mask.py 摄像头实时视频口罩佩戴检测演示
cnn.py 二分类网络结构定义


常见问题：
1）电脑上有多个摄像头
尝试把这一行 cap = cv2.VideoCapture(0)   0改为1 ， 就行了。

2）电脑没有摄像头怎么办？或摄像头不能正常工作，怎么办？
可以手机拍一段测试视频，传到电脑上，路径最好不要有中文。
把这一行cap = cv2.VideoCapture(0)
改为 cap = cv2.VideoCapture("d:\\20200305.mp4")
（假设你的视频文件放在d盘根目录下，名字为20200305.mp4，目录分隔符用\\，或写为r"d:\\20200305.mp4"）



附：人脸口罩数据集
链接：https://pan.baidu.com/s/1CCYnADFVnsXsBI1JFymsHg 
提取码：fvkb 
