import cv2

# 把一张图片嵌入另一张图片，并滑动
data1 = cv2.imread('img1.jpg')
cv2.imshow('origin', data1)
for i in range(data1.shape[0]):
    for j in range(data1.shape[1] // 2):
        t = data1[i, j, :]
        data1[i, j, :] = data1[i, data1.shape[1] - j - 1, :]
        data1[i, data1.shape[1] - j - 1, :] = t
cv2.imshow('img', data1)
cv2.waitKey()

#data[:100, data.shape[1] - 100:,:] = [0,255,255]
