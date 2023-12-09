import cv2
import numpy as np
head = cv2.imread('img2.jpg')
flag = cv2.imread('img1.jpg')
# 第6行为整体融合，第7~13行为渐变融合，可以二选一看看效果差别
result = head * 0.5 + flag * 0.5
result = np.zeros(head.shape)
for i in range(head.shape[0]):
    for j in range(head.shape[1]):
        r = (i + j) / (head.shape[1] // 2 + head.shape[0]// 2)
        if r > 1:
            r = 1
        result[i, j, :] = head[i, j, :] * r + flag[i, j, :] * (1 - r)
result = result.astype(np.uint8)
cv2.imshow('r', result)
cv2.waitKey()