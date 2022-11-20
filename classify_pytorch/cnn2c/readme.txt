用c语言实现深度学习推理。
搭建了一个简单的cnn网络，包含卷积层、ReLU、MaxPool、全连接层各一个，依次串联。

1. 先运行train_simple.py 进行手写字符识别训练。
2. 用tes_simple.py 进行测试，它会把网络参数保存为1.txt，可以打开这个文件看看有多少参数。
这里只输出了卷积层的参数。
3. 打开AI_course\case\mnist_c目录下的mnist_c.sln，在vs里运行程序，查看输出结果和python运行结果是否一致。
注：mnist.cpp里只实现了卷积层的c语言实现。