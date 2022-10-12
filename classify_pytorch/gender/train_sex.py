import torch
import math
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, models
import argparse
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class SexDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None):
        fh = open(txt, 'r')
        data = []
        for line in fh:
            line = line.strip('\n')
            words = line.split()
            data.append((float(words[0]) / 2.0, float(words[1]) / 80.0, int(words[2])))
        self.data = data

    def __getitem__(self, index):
        return torch.FloatTensor([self.data[index][0], self.data[index][1]]), self.data[index][2]

    def __len__(self):
        return len(self.data)

class SexNet(nn.Module):
    def __init__(self):
        super(SexNet, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(2, 2)
        )

    def forward(self, x):
        out = self.dense(x)
        return out

def train():
    os.makedirs('./output', exist_ok=True)
    train_data = SexDataset(txt='sex_train.txt')
    val_data = SexDataset(txt='sex_val.txt')
    batchsize = 10
    train_loader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batchsize)

    model = SexNet()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], 0.1)
    loss_func = nn.CrossEntropyLoss()

    epochs = 100
    for epoch in range(epochs):
        # training-----------------------------------
        model.train()
        train_loss = 0
        train_acc = 0
        for batch, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            print('epoch: %2d/%d batch %3d/%d  Train Loss: %.3f, Acc: %.3f'
                  % (epoch + 1, epochs, batch, math.ceil(len(train_data) / batchsize),
                     loss.item(), train_correct.item() / len(batch_x)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()  # 更新learning rate
        print('Train Loss: %.6f, Acc: %.3f' % (train_loss / (math.ceil(len(train_data)/batchsize)),
                                               train_acc / (len(train_data))))

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)

            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.item()
        print('Val Loss: %.6f, Acc: %.3f' % (eval_loss / (math.ceil(len(val_data)/batchsize)),
                                             eval_acc / (len(val_data))))
        # save model --------------------------------
        if (epoch + 1) % 1 == 0:
            # torch.save(model, 'output/model_' + str(epoch+1) + '.pth')
            torch.save(model.state_dict(), 'output/params_' + str(epoch + 1) + '.pth')
            #to_onnx(model, 3, 28, 28, 'params.onnx')

if __name__ == '__main__':
    train()
