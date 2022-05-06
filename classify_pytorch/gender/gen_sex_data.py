import numpy as np
import random

ftrain = open('sex_train.txt', 'wt')
fval = open('sex_val.txt', 'wt')

height = np.random.normal(1.80, 0.1, (100,))
weight = np.random.normal(75, 5, (100,))
for i in range(80):
    ftrain.write(f'{height[i]} {weight[i]} 1\n')
for i in range(80, 100):
    fval.write(f'{height[i]} {weight[i]} 1\n')

height = np.random.normal(1.62, 0.1, (100,))
weight = np.random.normal(50, 3, (100,))
for i in range(80):
    ftrain.write(f'{height[i]} {weight[i]} 0\n')
for i in range(80, 100):
    fval.write(f'{height[i]} {weight[i]} 0\n')

ftrain.close()
fval.close()

import matplotlib.pyplot as plt
import numpy as np
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


