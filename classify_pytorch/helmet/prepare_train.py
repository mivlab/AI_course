import os
import cv2

# 生成无头的人体图
def gen_no_head_image(dir):
    dstSize = 64
    files = os.listdir(dir)
    newdir = dir + '_1'
    os.makedirs(newdir, exist_ok=True)
    for file in files:
        img = cv2.imread(os.path.join(dir, file))
        if img is None:
            continue
        if img.shape[0] < img.shape[1] * 2: #仅对较长的图进行操作
            continue
        dstimg = cv2.resize(img[img.shape[0] - img.shape[1]:img.shape[0], :, :], (dstSize, dstSize))
        cv2.imwrite(os.path.join(newdir, file), dstimg)
        dstimg = cv2.resize(img[img.shape[0] // 2 - img.shape[1] // 2:img.shape[0] // 2 - img.shape[1] // 2 + img.shape[1], :, :], (dstSize, dstSize))
        cv2.imwrite(os.path.join(newdir, file[0:-4]+'_1.jpg'), dstimg)


# path:根目录, sub: 子目录
def gen_head_image(path, sub):
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

if __name__ == '__main__':
    gen_head_image(r'D:\data\rail\helmet\train', ['0_nohat', '0_nohat8000', '1_helmet'])
    #gen_no_head_image(r'D:\data\rail\helmet\train\0_nohat8000')