
1. 手写字符识别

1）下载数据集：链接：https://pan.baidu.com/s/18Fz9Cpj0Lf9BC7As8frZrw 提取码：xhgk

2）训练：train_mnist.py

需要加一个参数，--datapath=数据集所在的目录。

3）测试：tes_mnist.py 可以识别一张新的图片的类别。

2. 性别识别

给定一个人的身高和体重，预测ta的性别
代码位于gender目录下，运行train_sex.py 即可完成训练
gender\gender_with_annotation 目录是注释版代码，感谢许波波提供。


3. 人脸检测、人脸未戴口罩识别
在face目录下。


4. 图像分割
在seg目录下。


5. 冠状病毒肺炎X光识别

1）下载数据集  链接：https://pan.baidu.com/s/1XutIOtrt75GoBqq09KP3tw  提取码：nkuw 

2）训练：train_xchest.py 

需要加一个参数，--datapath=数据集所在的目录。

注意这个目录是images的上一级目录，因为train.csv里目录名称里已经带了images

代码里提供了两种神经网络模型，下面两行，采用其中之一：

model = Net112()

#model = models.resnet18(num_classes=4)  # 调用内置模型

Net112是手写字符识别相同的模型。resnet18是残差网络结构，效果更好，但计算量大一些。

3）测试：在tes_mnist.py上略作修改即可，自己实现。
 

常见问题：

1. 手写字符和肺炎X光的训练代码，有哪些地方不同？

答：你可以用文本比较工具（windows下可以装winmerge），比较train_mnist.py 和 train_xchest.py 两个文件，就知道改了哪些地方。

改了两点：

（1）train（）函数的开头，增加了读csv文件。因为肺炎图片的标签是用csv文件提供的，图片全部在同一目录，不像mnist数据是分目录存放的。代码如下：

    with open('train.csv') as csvfile:
	
        reader = csv.reader(csvfile)
		
        for i, line in enumerate(reader):
		
            if i > 0:
			
                f.write('%s %d\n' % (os.path.join(args.datapath, line[0]), int(line[1])))
				
				
（2）网络模型微小改动

model = Net112()	

你可以打开cnn.py，对比一下 Net112 和 Net两个类，差别在于全连接系数不一样：

nn.Linear(64 * 14 * 14, 128)  

nn.Linear(128, 4)  

其中，64x14x14是由输入图像大小112x112决定的， 4是预测类别数。






			
				
