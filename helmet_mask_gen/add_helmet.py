import cv2
import json
import os
import numpy as np
import math

def addHelmet(img, pt1, pt2, hatImg, hatPts):
    # dist: 原图上头的长度，帽子应占到一半，即0.5， dist1：帽子图上帽子的高度
    dist = math.sqrt((pt2[0] - pt1[0]) * (pt2[0] - pt1[0]) + (pt2[1] - pt1[1]) * (pt2[1] - pt1[1])) * 0.5
    mid = [(hatPts[1][0] + hatPts[2][0]) / 2, (hatPts[1][1] + hatPts[2][1]) / 2]
    dist1 = math.sqrt((hatPts[0][0] - mid[0]) * (hatPts[0][0] - mid[0]) + (hatPts[0][1] - mid[1]) * (hatPts[0][1] - mid[1]))

    # s为帽子图应缩小的比例
    s = dist / dist1
    hatImg = cv2.resize(hatImg, (int(hatImg.shape[1] * s), int(hatImg.shape[0] * s)))

    pts = np.array(hatPts)
    pts = pts * s # 帽子的关键点坐标也缩小同样尺度
    pts = pts.astype(np.int32)
    max0 = pts.max(0)
    min0 = pts.min(0)
    # rect: x, y, w, h。 矩形区域放大10个像素， todo：此处应考虑是否超出图像边界
    rect = [min0[0] - 10, min0[1] - 10, max0[0] - min0[0] + 20, max0[1] - min0[1] + 20]
    rectHat = hatImg[rect[1]:rect[3] + rect[1], rect[0]:rect[2] + rect[0]]

    # 矩形区域左上角与帽子顶部的相对位置
    relPos = [pts[0][0] - rect[0], pts[0][1] - rect[1]]

    start = [int(pt1[0] - relPos[0]), int(pt1[1] - relPos[1])]
    for y in range(start[1], start[1] + rect[3]):
        for x in range(start[0], start[0] + rect[2]):
            y1 = y - start[1]
            x1 = x - start[0]
            sum = int(rectHat[y1, x1, 0]) + int(rectHat[y1, x1, 1]) + int(rectHat[y1, x1, 2])
            if sum < 250 * 3: # 如果接近全白，为背景区域
                img[y, x, :] = rectHat[y1, x1, :]

if __name__ == "__main__":
    hatDir = 'onlyhat'
    hats = []
    for file in os.listdir(hatDir):
        if file[-3:] == 'jpg':
            hats.append(file)

    img = cv2.imread('1.jpg')

    with open("1.json", 'r') as load_f:
        load_dict = json.load(load_f)

    # 遍历帽子目录
    hatsPts = []
    for file in hats:
        f = open(os.path.join(hatDir, file[:-3]+'json'), 'r')
        load_dict1 = json.load(f)
        pt11 = load_dict['shapes'][0]['points']
        hatsPts.append(pt11)

    # 对原图中每一个人，叠加帽子
    for shapes in load_dict['shapes']:
        pt1 = shapes['points'][0]
        pt2 = shapes['points'][1]
        filename = '010.jpg' # 仅选一个帽子图片，作为示例
        hatImg = cv2.imread(os.path.join(hatDir, filename))
        f = open(os.path.join(hatDir, filename[:-3]+'json'), 'r')
        load_dict1 = json.load(f)
        pt11 = load_dict1['shapes'][0]['points']
        addHelmet(img, pt1, pt2, hatImg, pt11)

    cv2.imshow('out', img)
    cv2.waitKey()
