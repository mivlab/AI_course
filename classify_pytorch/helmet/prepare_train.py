import os
import cv2

if __name__ == '__main__':
    path = r'D:\data\rail\helmet\train' # 根目录
    sub = ['0_nohat'] #['0_nohat', '1_helmet'] # 子目录
    dstSize = 64
    for dir in sub:
        newdir = os.path.join(path, dir + '_1')
        os.makedirs(newdir, exist_ok=True)
        for file in os.listdir(os.path.join(path, dir)):
            img = cv2.imread(os.path.join(path, dir, file))
            if img is None:
                continue
            if img.shape[0] < img.shape[1]:
                continue
            dstimg = cv2.resize(img[0:img.shape[1], :, :], (dstSize, dstSize))
            cv2.imwrite(os.path.join(newdir, file), dstimg)

            # 要不要存水平镜像的图？
            #dstimg1 = cv2.flip(dstimg, 1)
            #cv2.imwrite(os.path.join(newdir, file[0:-4]+'_1.jpg'), dstimg1)

