import xml.etree.ElementTree as xmlDoc
import os
import cv2

imgDir = r'D:\data\face_mask\train'
faceOutDir = os.path.join(imgDir, 'face')
faceMaskOutDir = os.path.join(imgDir, 'face_mask')
os.makedirs(faceOutDir, exist_ok=True)
os.makedirs(faceMaskOutDir, exist_ok=True)

for file in os.listdir(imgDir):
    if file[-3:] == 'xml':
        path = os.path.join(imgDir, file)
        imagePath = os.path.join(imgDir, file[:-3]+'jpg')
        img = cv2.imread(imagePath)
        # root = xmlDoc.parse(os.path.join(imgDir, file))._root  # _root同getroot()
        root = xmlDoc.ElementTree(file=path).getroot()
        for i, node in enumerate(root.findall('object')):
            name = node.find('name').text
            box = node.find('bndbox')
            xmin = int(box.find('xmin').text)
            xmax = int(box.find('xmax').text)
            ymin = int(box.find('ymin').text)
            ymax = int(box.find('ymax').text)
            if name == 'face':
                fileName = os.path.join(faceOutDir, file[:-4]+'_%d.jpg'%(i))
            else:
                fileName = os.path.join(faceMaskOutDir, file[:-4]+'_%d.jpg'%(i))
            img1 = cv2.resize(img[ymin:ymax, xmin:xmax, :], (48, 64))
            cv2.imwrite(fileName, img1)
