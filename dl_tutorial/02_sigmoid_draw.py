import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#请修改代码，完成调用nn.Sigmoid()画出sigmoid曲线
m = nn.Sigmoid()
x = torch.arange(-10, 10, 0.1)
y = m(x)
xplot = x.numpy()
yplot = y.numpy()
plt.plot(xplot, yplot, color='#58b970', label='Regression Line')
plt.axis([-10, 10, 0, 1])
plt.show()