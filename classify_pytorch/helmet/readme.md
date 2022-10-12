未佩戴安全帽检测

1. prepare_train.py
用来准备训练数据。把多个目录下的行人上半身截取下来，resize为统一大小，
默认64x64。注意修改根目录和子目录名。
path  # 根目录
sub = ['0_nohat', '1_helmet'] # 子目录

运行的结果是生成新的目录，存放截好的图

2. train_helmet.py
可以开始训练了。训练过程和手写字符识别几乎一样。

3. tes_helmet.py
测试程序。


其他文件说明：
yolov5_video.cpp 这是在ncnn的yolov5例子基础上，把对单张图的检测，改为检测一段视频。
yolov5_video_save.cpp 这是用来读视频，把里面的人抠出来，保存。

