import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import os
from torch.utils.data import DataLoader #简化
from torch.utils.data import Dataset #简化

class SexDataset(Dataset):
    def __init__(self, txt):
        #载入数据
        fh = open(txt, 'r')
        data = []
        for line in fh:
            line = line.strip('\n')
            words = line.split()
            data.append((float(words[0]) / 2.0, float(words[1]) / 80.0, int(words[2])))
            #取出每行的 身高/2.0 体重/80.0 标签
            #这个步骤称为归一化，假设人的身高上限2.0，体重上限80，让绝大多数数据在[0,1]之间，有利于训练
        self.data = data

    def __getitem__(self, index):
        #读取数据
        return torch.FloatTensor([self.data[index][0], self.data[index][1]]), self.data[index][2]

    def __len__(self):
        #计算长度
        return len(self.data)

class SexNet(nn.Module):
    def __init__(self):
        super(SexNet, self).__init__()
        #向子类传输父类数据
        self.dense = nn.Sequential(
            nn.Linear(2, 2)
        )
        #构造神经网络模块
    def forward(self, x):
        #向前传输神经网络模块结果
        out = self.dense(x)
        return out

def train():
    os.makedirs('./output', exist_ok=True)
    train_data = SexDataset(txt='sex_train.txt')
    val_data = SexDataset(txt='sex_val.txt')
    batchsize = 10  #训练批次大小
    train_loader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batchsize)
    #加载训练数据 dataset 表示数据集  batch_size 表示数据集大小  shuffle 表示是否在加载时重新打乱数据
    model = SexNet()
    #简化函数名
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    # model.parameters() 返回迭代器  lr（learning rate）学习速率  weight_decay 权重衰减
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    # 初始学习速率lr是 0.01，在第10个epoch后变为 0.001，在第20个epoch后变为 0.0001
    # 一般来讲，要在前一级lr得到充分训练，即loss不降之后，再进入下一级lr的训练
    loss_func = nn.CrossEntropyLoss()
    #简化函数名 用以计算损失率，CrossEntropyLoss为交叉熵损失，是最常用的分类损失

    epochs = 100
    #进行 100 次训练
    for epoch in range(epochs):
        # training-----------------------------------
        model.train()
        #改变 行97中锁定状态  开始训练
        train_loss = 0
        train_acc = 0
        for batch, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            #取出 [身高,体重],男/女 的标签值
            out = model(batch_x)
            #进入神经网络 计算出分类结果
            loss = loss_func(out, batch_y)
            #计算损失率，用预测值out和标签值batch_y进行比较 (输出仍为 tensor() 类型)
            train_loss += loss.item()
            #将这个batch的损失率  加入每个 epoch 的总训练损失率当中。item()是对tensor类型变量取值，相当于类型转换
            pred = torch.max(out, 1)[1]
            #torch.max返回每个 [身高,体重] 分类结果中最大的值的索引  取出其索引对应的值
            #要了解torch.max的详细用法，推荐到pytorch官网文档搜索max， https://pytorch.org/docs/stable/index.html
            train_correct = (pred == batch_y).sum()
            #将所有分类结果与标签相同的值相加  (转化后仍为 tensor() 类型)
            train_acc += train_correct.item()
            #取出上述值 加入总正确个数
            print('epoch: %2d/%d batch %3d/%d  Train Loss: %.3f, Acc: %.3f'
                  % (epoch + 1, epochs, batch, math.ceil(len(train_data) / batchsize),
                     loss.item(), train_correct.item() / len(batch_x)))
            #打印
            #当前训练次数(epoch)编号、共多少训练次数、当前batch编号、共多少batch
            #当前batch损失率 当前batch正确率
            optimizer.zero_grad()
            #清空过往梯度
            loss.backward()
            #反向传播，计算当前梯度
            optimizer.step()
            #根据梯度更新网络参数
        scheduler.step()  # 更新学习速率
        print('Train Loss: %.6f, Acc: %.3f' % (train_loss / (math.ceil(len(train_data)/batchsize)),
                                               train_acc / (len(train_data))))
        #打印当前训练次数中 平均损失率 平均正确率
        model.eval()
        #对权值进行锁定 防止测试数据的输入影响 #用于小规模数据量时 减小测试数据影响
        eval_loss = 0
        eval_acc = 0
        #初始化
        for batch_x, batch_y in val_loader:
            #开始当前训练的正确率
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.item()
            # 68——81  CTRL+C CTRL+V 只改了储存变量
        print('Val Loss: %.6f, Acc: %.3f' % (eval_loss / (math.ceil(len(val_data)/batchsize)),
                                             eval_acc / (len(val_data))))
        #打印出当前训练模型对于测试数据的损失率与正确率
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 'output/params_' + str((epoch + 1)//10) + '.pth')
        #每进行10次训练 就保存一次训练模型

#这里是主程序开始
if __name__ == '__main__':
    train()
