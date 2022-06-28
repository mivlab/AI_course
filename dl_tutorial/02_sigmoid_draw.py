import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#请修改代码，完成调用nn.Sigmoid()画出sigmoid曲线
xplot = np.arange(-10, 10, 0.1)
yplot = np.sin(xplot)
plt.plot(xplot, yplot, color='#58b970', label='Regression Line')
plt.axis([-10, 10, -1, 1])
plt.show()