
import cv2
import json
import os
import numpy as np
import math

if __name__ == "__main__":
    maskImg = cv2.imread('mask2.png')
    img = cv2.imread('face.png')

    with open("mask2.json", 'r') as load_f:
        load_dict = json.load(load_f)
        pt = load_dict['shapes'][0]['points']
        mask_pt = load_dict['shapes'][1]['points']

    pt = np.array(pt, np.int32)
    mask_pt = np.array(mask_pt)
    face_pt = np.array([[83, 104], [140, 104]])
    d1 = math.sqrt((mask_pt[0][0] - mask_pt[1][0]) * (mask_pt[0][0] - mask_pt[1][0]) +\
        (mask_pt[0][1] - mask_pt[1][1]) + (mask_pt[0][1] - mask_pt[1][1]))
    d2 = math.sqrt((face_pt[0][0] - face_pt[1][0]) * (face_pt[0][0] - face_pt[1][0]) +\
        (face_pt[0][1] - face_pt[1][1]) + (face_pt[0][1] - face_pt[1][1]))
    s = d2 / d1

    # 两个眼睛与x轴的夹角，相减
    theta1 = math.asin((mask_pt[1][1] - mask_pt[0][1]) / d1)
    theta2 = math.asin((face_pt[1][1] - face_pt[0][1]) / d2)
    theta = theta1- theta2

    #旋转R和缩放s
    R = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
    R = s * R

    #a = math.cos(theta) * mask_pt[0][0] - math.sin(theta) * mask_pt[0][1]
    #b = math.cos(theta) * mask_pt[1][0] - math.sin(theta) * mask_pt[1][1]

    mask_pt = mask_pt.T

    mask_pt = np.matmul(R, mask_pt)

    # 平移
    tx = face_pt[0][0] - mask_pt[0][0]
    ty = face_pt[0][1] - mask_pt[1][0]

    # 生成掩码
    binImg = np.zeros((maskImg.shape[0],maskImg.shape[1]), np.uint8)
    cv2.fillPoly(binImg, [pt], (255))
    cv2.imshow('bin', binImg)

    # 填充
    for i in range(maskImg.shape[0]):
        for j in range(maskImg.shape[1]):
            if binImg[i, j] == 255:
                p = np.array([j, i])
                p1 = np.matmul(R, p) + np.array([tx, ty]).T
                p1 = p1.astype(np.uint32)
                img[p1[1], pt[0], :] = maskImg[i, j, :]

    cv2.imshow('mask', maskImg)
    cv2.imshow('image', img)
    cv2.waitKey()



