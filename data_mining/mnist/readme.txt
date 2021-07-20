这是基于paddlepaddle 的图像识别程序示例

mnist_clas.py 
手写字符识别程序，以jpg图像为输入。

mnist_clas_new.py 
手写字符识别，用DataLoader 实现。

zhubian_clas.py 
主变压器识别程序，二分类，单通道输入。

zhubian_clas_3c.py 
主变压器识别程序，二分类，三通道输入。

prepare_data.py 
根据接线图的标注信息（用labelme标注，生成json文件），自动生成训练用的正、负样本，分目录存放，用于分类器训练。图像位于image目录下。
