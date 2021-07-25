#coding=utf-8

# 双母线数据增强，重采样截图
import os
import numpy as np
import cv2
import random

#coding=utf-8
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img

def cv_imwrite(file_path, img):
    cv2.imencode('.jpg', img)[1].tofile(file_path)
    #cv2.imencode(np.fromfile(file_path,dtype=np.uint8), img)


#dir = r'D:\data\electr\datamining\接线图\分电压截图\双母线双分段'
#dir = r'D:\data\electr\datamining\接线图\分电压截图\双母线'
dir = r'D:\data\electr\datamining\接线图\分电压截图\单母线'
out_dir = os.path.join(dir, '1')
os.makedirs(out_dir, exist_ok=True)
for j, file in enumerate(os.listdir(dir)):
    if file[-3:] != 'jpg' and file[-3:] != 'png':
        continue
    name = os.path.join(dir, file)
    img = cv_imread(name)
    if img is None:
        print(name, 'does not exist.')
        continue
    h, w, _ = img.shape
    cx = w // 2
    cy = h // 2
    for i in range(2000):
        bx1 = random.randint(h * 2 // 3, h)
        x1 = random.randint(cx - bx1 // 4, cx + bx1 // 4) - bx1 // 2
        y1 = random.randint(cy - bx1 // 4, cy + bx1 // 4) - bx1 // 2
        if x1 + bx1 > w or x1 < 0 or y1 + bx1 > h or y1 < 0:
            continue
        out = os.path.join(out_dir, str(j) + '_' + str(i) + '.jpg')
        img1 = cv2.resize(img[y1: y1 + bx1, x1: x1 + bx1, :], (256, 256))
        cv_imwrite(out, img1)