import numpy as np
#import random
import gender_plot


#生成性别有关 身高 体重 的数据
ftrain = open('sex_train.txt', 'wt')
fval = open('sex_val.txt', 'wt')

#np.random.normal() 参数 1:均值 2:偏差值 3:生成数据量
height = np.random.normal(1.80, 0.1, (100,))
weight = np.random.normal(75, 5, (100,))
#规定男性数据生成方法
for i in range(80):
    ftrain.write(f'{height[i]} {weight[i]} 1\n')
for i in range(80, 100):
    fval.write(f'{height[i]} {weight[i]} 1\n')
#生成男性数据

height = np.random.normal(1.62, 0.1, (100,))
weight = np.random.normal(50, 3, (100,))
#规定女性数据生成方法
for i in range(80):
    ftrain.write(f'{height[i]} {weight[i]} 0\n')
for i in range(80, 100):
    fval.write(f'{height[i]} {weight[i]} 0\n')
#生成女性数据

ftrain.close()
fval.close()


#用以生成数据的直观图
gender_plot.show()

