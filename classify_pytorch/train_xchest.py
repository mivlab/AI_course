import torch
import math
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, models
import argparse
import os
from torch.utils.data import DataLoader

from dataloader import mnist_loader as ml
from models.cnn import Net
from models.cnn import Net112
import csv

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
parser.add_argument('--use_cuda', default=False, help='using CUDA for training')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
if args.cuda:
    torch.backends.cudnn.benchmark = True


def train():
    os.makedirs('./output', exist_ok=True)
    f = open('output/total.txt', 'wt')
    with open('train.csv') as csvfile:
        reader = csv.reader(csvfile)
        for i, line in enumerate(reader):
            if i > 0:
                f.write('%s %d\n' % (os.path.join(args.datapath, line[0]), int(line[1])))
    f.close()
    ml.shuffle_split('output/total.txt', 'output/train.txt', 'output/val.txt')

    train_data = ml.MyDataset(txt='output/train.txt', transform=transforms.ToTensor())
    val_data = ml.MyDataset(txt='output/val.txt', transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size)

    model = Net112()
    #model = models.resnet18(num_classes=4)  # 调用内置模型
    #model.load_state_dict(torch.load('./output/params_10.pth'))

    if args.cuda:
        print('training with cuda')
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        # training-----------------------------------
        model.train()
        train_loss = 0
        train_acc = 0
        for batch, (batch_x, batch_y) in enumerate(train_loader):
            if args.cuda:
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)  # 256x3x28x28  out 256x10
            loss = loss_func(out, batch_y)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            print('epoch: %2d/%d batch %3d/%d  Train Loss: %.3f, Acc: %.3f'
                  % (epoch + 1, args.epochs, batch, math.ceil(len(train_data) / args.batch_size),
                     loss.item(), train_correct.item() / len(batch_x)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()  # 更新learning rate
        print('Train Loss: %.6f, Acc: %.3f' % (train_loss / (math.ceil(len(train_data)/args.batch_size)),
                                               train_acc / (len(train_data))))

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for batch_x, batch_y in val_loader:
            if args.cuda:
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)

            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.item()
        print('Val Loss: %.6f, Acc: %.3f' % (eval_loss / (math.ceil(len(val_data)/args.batch_size)),
                                             eval_acc / (len(val_data))))
        # save model --------------------------------
        if (epoch + 1) % 1 == 0:
            # torch.save(model, 'output/model_' + str(epoch+1) + '.pth')
            torch.save(model.state_dict(), 'output/params_' + str(epoch + 1) + '.pth')


if __name__ == '__main__':
    train()
