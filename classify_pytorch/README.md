
1. 手写字符识别
1）下载数据集：https://pan.baidu.com/s/1X-FB-SKUvVvWkXdo_b8SHA, password：mu8h
2）训练：train_mnist.py
需要加一个参数，--datapath=数据集所在的目录。
3）测试：test_mnist.py 可以识别一张新的图片的类别。

2. 冠状病毒肺炎X光识别
1）下载数据集
2）训练：train_xchest.py 
需要加一个参数，--datapath=数据集所在的目录。
注意这个目录是images的上一级目录，因为train.csv里目录名称里已经带了images
代码里提供了两种神经网络模型，下面两行，采用其中之一：
model = Net112()
#model = models.resnet18(num_classes=4)  # 调用内置模型
Net112是手写字符识别相同的模型。resnet18是残差网络结构，效果更好，但计算量大一些。
3）测试：可以在test_mnist.py上略作修改。
 
