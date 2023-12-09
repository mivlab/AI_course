# 打开一张图片，把左上角100x100像素区域修改为黑色
import cv2
data = cv2.imread('img2.jpg')
data[0:100, 0:100, :] = 0
# 或写作下面形式，填充为黄色
#data[0:100, 0:100, :] = (0, 255, 255)
cv2.imshow('img', data)
cv2.waitKey()