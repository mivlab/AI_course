#coding:utf-8
import cv2
import os
import json
import numpy as np

# 图像根目录
dataPath = r'F:\data\electr\jueyuanzi\my_data'

for file in os.listdir(dataPath): # 遍历根目录下每个图像
    if file[-3:] == 'jpg' or file[-3:] == 'png':
        imageFile = os.path.join(dataPath, file)
        #image = cv2.imread(os.path.join(dataPath, file)) # 此函数不支持中文文件名，因此用下面一行替代
        image = cv2.imdecode(np.fromfile(imageFile, dtype=np.uint8), -1)
        if image is None: # 如果读图片失败，则跳过
            continue
        cv2.imshow('image', image) # 显示图片

        w = image.shape[1] # 图像宽
        h = image.shape[0] # 图像高
        center = (image.shape[1] // 2, image.shape[0] // 2) # 旋转中心
        M = cv2.getRotationMatrix2D(center, 45, 1.0) # M为旋转矩阵2x3
        rotated = cv2.warpAffine(image, M, (w, h)) # 进行仿射变换

        jsonFile = os.path.join(dataPath, file[:-3] + 'json')
        if not os.path.exists(jsonFile): # 如果文件不存在，则跳过
            continue
        with open(jsonFile, 'r', encoding="utf-8") as load_f:
            load_dict = json.load(load_f)
            for object in load_dict['shapes']: # 此处可以设断点调试，看看load_dict、object这些变量的类型和值
                pts = np.zeros((len(object['points']), 2), dtype=np.float32)
                for i, point in enumerate(object['points']):
                    point.append(1.0)
                    pt = np.array(point, np.float32).reshape((3, 1))
                    pt1 = np.matmul(M, pt) # 矩阵乘法
                    pts[i, :] = pt1.reshape(2)
                    cv2.circle(rotated, (int(pt1[0]), int(pt1[1])), 3, (0, 0, 255), -1) # 画实心圆
                p1 = pts.astype(dtype=np.int32).min(0)
                p2 = pts.astype(dtype=np.int32).max(0)
                cv2.rectangle(rotated, p1, p2, (0, 255, 0), 1)
        cv2.imshow('rotated', rotated) # 显示旋转后的图像
        cv2.waitKey()
