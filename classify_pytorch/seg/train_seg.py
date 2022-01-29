import torch
import math
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, models
import argparse
import os
from torch.utils.data import DataLoader
import sys
#sys.path.append("..")
from torch.utils.data import Dataset
import cv2
from PIL import Image
import numpy as np
import torch.nn.functional as F

class SegDataset(Dataset):
    def __init__(self, root=r'D:\data\VOCdevkit\VOC2012', file='train.txt', w=144, h=144, transform=None):
        fh = open(os.path.join(root, 'ImageSets', 'Segmentation', file), 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            imgs.append((os.path.join(root, 'JPEGImages', line + '.jpg'), os.path.join(root, 'SegmentationClass', line + '.png')))
        self.imgs = imgs
        self.transform = transform
        self.width = w
        self.height = h

    def __getitem__(self, index):
        imagename, labelname = self.imgs[index]
        img = cv2.imread(imagename, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.width, self.height))
        if self.transform is not None:
            img = self.transform(img)
        mask = Image.open(labelname)
        mask = mask.resize((self.width, self.height), Image.NEAREST)
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return img, torch.from_numpy(target).long()

    def __len__(self):
        return len(self.imgs)


class Net_seg(nn.Module):
    def __init__(self):
        super(Net_seg, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  #
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 32x14x14
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  #
            nn.ReLU(),
            nn.MaxPool2d(2)  #
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),  #
            nn.ReLU(),
            nn.MaxPool2d(2)  #
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 21, 3, 1, 1)  #
        )

    def forward(self, x):
        size = x.size()[2:]
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        head = self.head(conv3_out)
        out = F.interpolate(head, size, mode='bilinear', align_corners=True)
        return out


def train():
    os.makedirs('./output', exist_ok=True)
    width = 128
    height = 128
    train_data = SegDataset(root=r'D:\data\segmentation\VOCdevkit\VOC2012', file='train.txt', w=width, h=height, transform=transforms.ToTensor())
    val_data = SegDataset(root=r'D:\data\segmentation\VOCdevkit\VOC2012', file='val.txt', w=width, h=height, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size)

    model = Net_seg()
    #model = models.resnet18(num_classes=10)  # 调用内置模型
    #model.load_state_dict(torch.load('./output/params_2.pth'))

    if args.cuda:
        print('training with cuda')
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    loss_func = nn.CrossEntropyLoss(ignore_index=-1)

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
            train_correct = (pred == batch_y).sum() / (width * height)
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
            out = out.squeeze()
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum() / (width * height)
            eval_acc += num_correct.item()
        print('Val Loss: %.6f, Acc: %.3f' % (eval_loss / (math.ceil(len(val_data)/args.batch_size)),
                                             eval_acc / (len(val_data))))
        # save model --------------------------------
        if (epoch + 1) % 1 == 0:
            # torch.save(model, 'output/model_' + str(epoch+1) + '.pth')
            torch.save(model.state_dict(), 'output/params_' + str(epoch + 1) + '.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--use_cuda', default=True, help='using CUDA for training')

    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.backends.cudnn.benchmark = True
    train()
