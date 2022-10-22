import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from train_sex import SexNet
# 保证图片在浏览器内正常显示
#%matplotlib inline

ftrain = open('sex_train.txt', 'rt')
fval = open('sex_val.txt', 'rt')
male = []
female = []
for line in ftrain.readlines():
    word = line.strip().split()
    if int(word[2]) == 1:
        male.append([float(word[0]), float(word[1])])
    else:
        female.append([float(word[0]), float(word[1])])

male = np.array(male)
female = np.array(female)

plt.scatter(male[:,0], male[:,1], marker='o')
plt.scatter(female[:,0], female[:,1], marker='^')

plt.show()



