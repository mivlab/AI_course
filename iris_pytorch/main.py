import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# 逻辑回归
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(4, 3)

    def forward(self, x):
        x = self.lr(x)
        return x

# 多层感知机
class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        X = self.softmax(X)

        return X

if __name__ == '__main__':
    # load IRIS dataset
    dataset = pd.read_csv('dataset/iris.csv')

    # transform species to numerics
    dataset.loc[dataset.species=='Iris-setosa', 'species'] = 0
    dataset.loc[dataset.species=='Iris-versicolor', 'species'] = 1
    dataset.loc[dataset.species=='Iris-virginica', 'species'] = 2

    train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:4]].values,
                                                        dataset.species.values, test_size=0.1)

    # wrap up with Variable in pytorch
    train_X = Variable(torch.Tensor(train_X).float())
    test_X = Variable(torch.Tensor(test_X).float())
    train_y = Variable(torch.Tensor(train_y).long())
    test_y = Variable(torch.Tensor(test_y).long())


    net = LogisticRegression()
    #net = Net()
    criterion = nn.CrossEntropyLoss()# cross entropy loss

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(1000):
        net.train()
        optimizer.zero_grad()
        out = net(train_X)
        _, predict_y = torch.max(out, 1)
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('number of epoch', epoch, 'loss', loss.item(), 'acc ', accuracy_score(train_y.data, predict_y.data))

    net.eval()
    predict_out = net(test_X)
    _, predict_y = torch.max(predict_out, 1)


    acc = accuracy_score(test_y.data, predict_y.data)
    p = precision_score(test_y.data, predict_y.data, average='macro')
    r = recall_score(test_y.data, predict_y.data, average='macro')

    print('accuracy', acc, ',macro precision', p, ',macro recall', r)
