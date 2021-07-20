import json
import cv2
import os
import random

# 正样本
files = ['1.json', '2.json', '4.json']
dir = r'image'
out_dir = os.path.join(dir, '0')
os.makedirs(out_dir, exist_ok=True)
for file in files:
    name = file[0:-4] + 'bmp'
    img = cv2.imread(os.path.join(dir, name))
    if img is None:
        print(name, 'does not exist.')
    f = open(os.path.join(dir, file), 'r')
    load_dict = json.load(f)
    for j, rect in enumerate(load_dict['shapes']):
        xmin = int(rect['points'][0][0])
        ymin = int(rect['points'][0][1])
        xmax = int(rect['points'][1][0])
        ymax = int(rect['points'][1][1])
        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2
        bx = xmax - xmin + 1
        by = ymax - ymin + 1
        box = int(max(bx, by) * 1.2)
        for i in range(500):
            cx1 = cx + random.randint(-box // 6, box // 6)
            cy1 = cy + random.randint(-box // 6, box // 6)
            bx1 = box + random.randint(-box // 6, box // 6)
            out = os.path.join(out_dir, file[0:-5] + '_' + str(j) + '_' + str(i) + '.jpg')
            img1 = cv2.resize(img[cy1 - bx1 // 2: cy1 + bx1 // 2, cx1 - bx1 // 2: cx1 + bx1 // 2, :], (32, 32))
            cv2.imwrite(out, img1)


# 负样本
out_dir = os.path.join(dir, '1')
os.makedirs(out_dir, exist_ok=True)
for file in files:
    name = file[0:-4] + 'bmp'
    img = cv2.imread(os.path.join(dir, name))
    if img is None:
        print(name, 'does not exist.')
    f = open(os.path.join(dir, file), 'r')
    load_dict = json.load(f)
    for i in range(1500):
        bx1 = random.randint(30, 50)
        cx1 = random.randint(bx1, img.shape[1] - bx1)
        cy1 = random.randint(bx1, img.shape[0] - bx1)
        out = os.path.join(out_dir, file[0:-5] + '_' + str(i) + '.jpg')
        img1 = cv2.resize(img[cy1 - bx1 // 2: cy1 + bx1 // 2, cx1 - bx1 // 2: cx1 + bx1 // 2, :], (32, 32))
        cv2.imwrite(out, img1)
