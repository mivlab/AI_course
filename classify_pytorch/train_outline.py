import torch
import math
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    def __len__(self):
        return len(self.data)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Sequential(nn.Linear(in_channel, 10))
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

def train():
    use_cuda = torch.cuda.is_available()
    train_data = MyDataset(data, label, transform=transforms.ToTensor())
    val_data = MyDataset(data, label, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=128)
    model = Net()
    #model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(30):
        # training-----------------------------------
        model.train()
        for batch, (batch_x, batch_y) in enumerate(train_loader):
            #batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)  # 256x3x28x28  out 256x10
            loss = loss_func(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()  # 更新learning rate
        # evaluation--------------------------------
        model.eval()
        for batch_x, batch_y in val_loader:
            #batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
        # save model --------------------------------
        torch.save(model.state_dict(), 'params_' + str(epoch + 1) + '.pth')

if __name__ == '__main__':
    train()
