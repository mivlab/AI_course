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



