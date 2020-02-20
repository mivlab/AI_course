
import cv2
img = cv2.imread('test.jpg')

# 图像resize
img1 = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
cv2.imshow("img1", img1)

# 取图像区域
img2 = img[0:img.shape[0], 0:img.shape[1] // 2]  # 取左半部分
cv2.imshow("img2", img2)

# 修改图像
crop_img = img[100:300, 50:250]
img[100:300, 50:250] = (0, 0, 0)
cv2.imshow("crop_img", crop_img)
cv2.imshow("img", img)

cv2.waitKey(0)

# 打开视频

cap = cv2.VideoCapture('test.avi')
if not cap.isOpened():
    print("Error opening video file")
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    cv2.imshow('frame', frame)
    cv2.waitKey(100)
