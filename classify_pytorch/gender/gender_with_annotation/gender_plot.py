import matplotlib.pyplot as plt
import numpy as np

def show():
    ftrain = open('sex_train.txt', 'rt')
    fval = open('sex_val.txt', 'rt')
    #打开 txt 文件 用以储存数据
    male = []#男性数据
    female = []#女性数据
    for line in ftrain.readlines():
        #按行读取
        word = line.strip().split()
        #拆分
        if int(word[2]) == 1:
            #若男性 将身高体重添加至 male
            male.append([float(word[0]), float(word[1])])
        else:
            #若女性 将身高体重添加至 female
            female.append([float(word[0]), float(word[1])])

    male = np.array(male)
    female = np.array(female)
    #将 male female 转化 array类型

    plt.scatter(male[:,0], male[:,1], marker='o')
    plt.scatter(female[:,0], female[:,1], marker='^')
    #添加图片的数据  (颜色自动生成)
    plt.show()
    #显示图片