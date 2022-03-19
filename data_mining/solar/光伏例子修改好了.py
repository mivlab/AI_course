import paddle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import tqdm

data = pd.read_csv('./dataset.csv')
# 如果要增减特征数量，只需修改此处即可
data_new = data.drop(columns=["timeStamp","windsp", "winddir"])#'air_temp',
data_new.head()

data_ = data_new.values
input_dim = data_.shape[1] # 输入特征维度

# 按比例划分训练集、测试集
split_boundary = int(data_.shape[0] * 0.8)
train = data_[: split_boundary]
test = data_[split_boundary:]

# 归一化
mean = train.mean(axis=0)
std = train.std(axis=0)
train = (train - mean) / std
test = (test - mean) / std

time_steps = 60
output_steps = 1 # 可以预测一步，或预测多步
target_index = 0 # 待预测变量是第几个特征

class MyDataset(paddle.io.Dataset):
    """
    data: 输入数据，格式为 时间长度 x 特征维度
    time_steps: 输入时间步长
    output_steps： 预测时间步长
    target_index:  待预测的变量是第几个特征
    """
    def __init__(self, data, time_steps, output_steps, target_index):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        注意：这个是不需要label
        """
        super(MyDataset, self).__init__()
        self.time_steps = time_steps
        self.output_steps = output_steps
        self.target_index = target_index
        self.data = self.transform(data.astype(np.float32))

    def transform(self, data):
        '''
        构造时序数据
        '''
        output = []
        for i in range(data.shape[0] - self.time_steps - self.output_steps):
            output.append(data[i: (i + self.time_steps + self.output_steps), :])#.transpose((1, 0))
        return np.stack(output)

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据）
        """
        data = self.data[index, 0:self.time_steps, :]
        label = self.data[index, self.time_steps:, target_index]
        return data, label

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.data)


train_dataset = MyDataset(train, time_steps, output_steps, target_index)
test_ds = MyDataset(test, time_steps, output_steps, target_index)


# 定义LSTM网络
class MyLSTMModel(paddle.nn.Layer):
    '''
    DNN网络
    '''

    def __init__(self):
        super(MyLSTMModel, self).__init__()
        # LSTM 三个参数依次为：输入特征维度、隐藏层节点数、网络层数（为加快速度，层数少一点）
        self.rnn = paddle.nn.LSTM(input_dim, 16, 1)
        self.flatten = paddle.nn.Flatten()
        self.fc1 = paddle.nn.Linear(16 * time_steps, 120) # lstm的输出是隐藏层节点个数乘以时间维度
        self.relu = paddle.nn.PReLU()
        self.fc2 = paddle.nn.Linear(120, output_steps)

    def forward(self, input):  # forward 定义执行实际运行时网络的执行逻辑
        '''前向计算'''
        # print('input',input.shape)
        out, (h, c) = self.rnn(input)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 参数设置
epoch_num = 20
batch_size = 256
learning_rate = 0.001


def train():
    print('训练开始')
    # 实例化模型
    model = MyLSTMModel()
    # 将模型转换为训练模式
    model.train()
    # 设置优化器，学习率，并且把模型参数给优化器
    opt = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())
    # 设置损失函数
    mse_loss = paddle.nn.MSELoss()
    # 设置数据读取器
    data_reader = paddle.io.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       drop_last=True
                                       )

    history_loss = []
    iter_epoch = []
    for epoch in range(epoch_num):
        for data, label in data_reader():
            train_ds = data.reshape((batch_size, time_steps, input_dim))
            train_lb = label.reshape((batch_size, output_steps))
            out = model(train_ds)
            avg_loss = mse_loss(out, train_lb)
            avg_loss.backward()
            opt.step()
            opt.clear_grad()
        print('epoch {}, loss {}'.format(epoch, avg_loss.numpy()[0]))
        iter_epoch.append(epoch)
        history_loss.append(avg_loss.numpy()[0])
    # 绘制loss
    plt.plot(iter_epoch, history_loss, label='loss')
    plt.legend()
    plt.xlabel('iters')
    plt.ylabel('Loss')
    plt.show()
    # 保存模型参数
    paddle.save(model.state_dict(), 'model')


train()

param_dict = paddle.load('model')  # 读取保存的参数
model = MyLSTMModel()
model.load_dict(param_dict)  # 加载参数
model.eval()  # 预测
data_reader1 = paddle.io.DataLoader(test_ds,
                                    places=[paddle.CPUPlace()],
                                    batch_size=batch_size,
                                    )
res = []
res1 = []

for data, label in data_reader1():
    data = data.reshape((batch_size, time_steps, input_dim))
    label = label.reshape((batch_size, output_steps))
    out = model(data)
    res.extend(out.numpy().reshape(batch_size).tolist())
    res1.extend(label.numpy().reshape(batch_size).tolist())


title = "t321"
plt.title(title, fontsize=24)
plt.xlabel("time", fontsize=14)
plt.ylabel("irr", fontsize=14)
plt.plot(res, color='red', label='predict')
plt.plot(res1, color='g', label='real')
plt.legend()
plt.grid()
plt.show()
