import cv2
import numpy as np
image1 = cv2.imread('img1.jpg')
image2 = cv2.imread('img2.jpg')
#image = image1 * 0.3 + image2 * 0.7
image = np.zeros(image1.shape)
for i in range(image1.shape[0]):
    for j in range(image1.shape[1]):
        r = j / image1.shape[1]
        image[i, j, :] = image1[i, j, :] * (1-r) + image2[i, j, :] * r
image = image.astype(np.uint8)
cv2.imshow('img1', image1)
cv2.imshow('img2', image2)
cv2.imshow('img', image)
cv2.waitKey()