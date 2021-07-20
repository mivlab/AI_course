#加载飞桨、Numpy和相关类库
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import random
from PIL import Image

WIDTH = 32
HEIGHT = 32
#datapath = r'D:\data\MNIST_Dataset\train_images' # 训练样本目录
datapath = r'D:\proj\python\images\graph_part'

# 生成训练图像列表
def image_list(imageRoot, txt='list.txt'):
    f = open(txt, 'wt')
    for (label, filename) in enumerate(sorted(os.listdir(imageRoot), reverse=False)):
        if os.path.isdir(os.path.join(imageRoot, filename)):
            for imagename in os.listdir(os.path.join(imageRoot, filename)):
                name, ext = os.path.splitext(imagename)
                ext = ext[1:]
                if ext == 'jpg' or ext == 'png' or ext == 'bmp':
                    f.write('%s %d\n' % (os.path.join(imageRoot, filename, imagename), label))
    f.close()

# 打开图像
def load_image(file):
    im = Image.open(file).convert('L')                        #将RGB转化为灰度图像，L代表灰度图像，像素值在0~255之间
    im = im.resize((HEIGHT, WIDTH), Image.ANTIALIAS)          #图像缩放为统一大小
    im = np.array(im)
    #im = im.transpose((2, 0, 1)).astype(np.float32)
    im = im.reshape(1, 1, HEIGHT, WIDTH).astype(np.float32) #返回新形状的数组,把它变成一个 numpy 数组以匹配数据馈送格式。
    im = im / 255.0 * 2.0 - 1.0                               #归一化到[-1~1]之间
    return im

# 加载训练数据
def load_data(datapath):
    os.makedirs('./output', exist_ok=True)
    image_list(datapath, 'output/total.txt')
    f = open('output/total.txt', 'r')
    lines = f.readlines()
    random.shuffle(lines)
    data = np.zeros((len(lines), 1, HEIGHT, WIDTH), dtype=np.float32)
    label = np.zeros(len(lines), dtype=np.int64)
    for i, line in enumerate(lines):
        name, label_ = line.strip().split()
        img = load_image(name)
        data[i, :, :, :] = img
        label[i] = int(label_)

    # 训练集和测试集的划分比例
    offset = int(len(lines) * 0.8)
    training_data = data[:offset, :, :, :]
    training_label = label[:offset]
    test_data = data[offset:, :, :, :]
    test_label = label[:offset]
    return training_data, training_label, test_data, test_label

# 定义卷积神经网络
class Net28(paddle.nn.Layer):
    # self代表类的实例自身
    def __init__(self):
        # 初始化父类中的一些参数
        super(Net28, self).__init__()

        # 定义网络组件
        self.conv1 = paddle.nn.Conv2D(1, 16, 3, 1, 1)
        self.relu = paddle.nn.PReLU()
        self.pool1 = paddle.nn.MaxPool2D(2, 2)
        self.conv2 = paddle.nn.Conv2D(16, 32, 3, 1, 1)
        self.pool2 = paddle.nn.MaxPool2D(2, 2)
        self.fc1 = Linear(in_features= 7 * 7 * 32, out_features=128)
        self.fc2 = Linear(in_features= 128, out_features=10)

    # 网络的前向计算
    def forward(self, inputs):
        conv1_out = self.relu(self.conv1(inputs))
        pool1_out = self.pool1(conv1_out)
        conv2_out = self.relu(self.conv2(pool1_out))
        pool2_out = self.pool1(conv2_out)
        d = pool2_out.reshape((pool2_out.shape[0], 7 * 7 * 32))
        x1 = self.fc1(d)
        x = self.fc2(x1)
        return x

class Net32(paddle.nn.Layer):
    # self代表类的实例自身
    def __init__(self):
        # 初始化父类中的一些参数
        super(Net32, self).__init__()

        # 定义网络组件
        self.conv1 = paddle.nn.Conv2D(1, 16, 3, 1, 1)
        self.relu = paddle.nn.PReLU()
        self.pool1 = paddle.nn.MaxPool2D(2, 2)
        self.conv2 = paddle.nn.Conv2D(16, 32, 3, 1, 1)
        self.pool2 = paddle.nn.MaxPool2D(2, 2)
        self.fc1 = Linear(in_features= 8 * 8 * 32, out_features=128)
        self.fc2 = Linear(in_features= 128, out_features=2)

    # 网络的前向计算
    def forward(self, inputs):
        conv1_out = self.relu(self.conv1(inputs))
        pool1_out = self.pool1(conv1_out)
        conv2_out = self.relu(self.conv2(pool1_out))
        pool2_out = self.pool2(conv2_out)
        d = pool2_out.reshape((pool2_out.shape[0], 8 * 8 * 32))
        x1 = self.fc1(d)
        x = self.fc2(x1)
        return x

# 声明定义好的线性回归模型
model = Net32()
# 开启模型训练模式
model.train()
# 加载数据
training_data, training_label, test_data, test_label = load_data(datapath)
# 定义优化算法，使用随机梯度下降SGD，学习率设置为0.01
opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())

EPOCH_NUM = 5  # 设置外层循环次数
BATCH_SIZE = 128  # 设置batch大小

# 定义外层循环
for epoch_id in range(EPOCH_NUM):
    start = 0
    # 定义内层循环
    for iter_id in range(0, len(training_data), BATCH_SIZE):
        x = np.array(training_data[start:start + BATCH_SIZE, :, :, :])  # 获得当前批次训练数据
        y = np.array(training_label[start:start + BATCH_SIZE])  # 获得当前批次训练标签（真实房价）
        # 将numpy数据转为飞桨动态图tensor形式
        x = paddle.to_tensor(x)
        y = paddle.to_tensor(y)
        # 前向计算
        predicts = model(x)
        # 计算损失
        loss = F.cross_entropy(predicts, label = y)
        avg_loss = paddle.mean(loss)
        if iter_id % 20 == 0:
            print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))

        # 反向传播
        avg_loss.backward()
        # 最小化loss,更新参数
        opt.step()
        # 清除梯度
        opt.clear_grad()
        start = start + BATCH_SIZE

# 保存模型参数，文件名为LR_model.pdparams
paddle.save(model.state_dict(), 'LR_model.pdparams')
print("模型保存成功，模型参数保存在LR_model.pdparams中")

# 第2题
model.eval()
tp = 0
fp = 0
p = []
testpath = r'D:\proj\python\images\test100\1'
for name in os.listdir(testpath):
    img = load_image(os.path.join(testpath, name))
    x = paddle.to_tensor(img)
    predicts = model(x)
    p = predicts.numpy().argmax() # 最大索引
    if p == 0:
        fp += 1

testpath = r'D:\proj\python\images\test100\0'
for name in os.listdir(testpath):
    img = load_image(os.path.join(testpath, name))
    x = paddle.to_tensor(img)
    predicts = model(x)
    p = predicts.numpy().argmax() # 最大索引
    if p == 0:
        tp += 1

print('recall=', tp / 50.0)
print('fpr=', fp / 50.0)



# 第3题

model.eval()
tp = 0
fp = 0
pos = []
neg = []
testpath = r'D:\proj\python\images\test100\0'
for name in os.listdir(testpath):
    img = load_image(os.path.join(testpath, name))
    x = paddle.to_tensor(img)
    predicts = model(x)
    out = F.softmax(predicts)
    out = out.numpy()
    pos.append(out[0][0])


testpath = r'D:\proj\python\images\test100\1'
for name in os.listdir(testpath):
    img = load_image(os.path.join(testpath, name))
    x = paddle.to_tensor(img)
    predicts = model(x)
    out = F.softmax(predicts)
    out = out.numpy()
    neg.append(out[0][0])

recall = []
fpr = []
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    tp = 0
    fp = 0
    for p in pos:
        if p > t:
            tp += 1
    for n in neg:
        if n > t:
            fp += 1
    recall.append(tp / 50)
    fpr.append(fp / 50)

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(fpr, recall, '-')
plt.xlabel('false positive rate')
plt.ylabel('recall')
plt.show()
