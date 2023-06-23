from yolov3_example import TrainDataset
from yolov3_example import TRAINDIR
from yolov3_example import TESTDIR
from yolov3_example import VALIDDIR
from yolov3_example import YOLOv3
from yolov3_example import get_lr
from yolov3_example import NUM_CLASSES
from yolov3_example import ANCHORS
from yolov3_example import ANCHOR_MASKS
from yolov3_example import IGNORE_THRESH
from yolov3_example import bsize

import time
import paddle
import numpy as np

if __name__ == '__main__':

    #TRAINDIR = '/home/aistudio/work/insects/train'
    #TESTDIR = '/home/aistudio/work/insects/test'
    #VALIDDIR = '/home/aistudio/work/insects/val'
    #paddle.set_device("gpu:0")
    # 创建数据读取类
    train_dataset = TrainDataset(TRAINDIR, mode='train')
    valid_dataset = TrainDataset(VALIDDIR, mode='valid')
    test_dataset = TrainDataset(VALIDDIR, mode='valid')
    # 使用paddle.io.DataLoader创建数据读取器，并设置batchsize，进程数量num_workers等参数
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=bsize, shuffle=True, num_workers=0, drop_last=True, use_shared_memory=False)
    valid_loader = paddle.io.DataLoader(valid_dataset, batch_size=bsize, shuffle=False, num_workers=0, drop_last=False, use_shared_memory=False)
    model = YOLOv3(num_classes = NUM_CLASSES)  #创建模型
    learning_rate = get_lr()
    opt = paddle.optimizer.Momentum(
                 learning_rate=learning_rate,
                 momentum=0.9,
                 weight_decay=paddle.regularizer.L2Decay(0.0005),
                 parameters=model.parameters())  #创建优化器
    # opt = paddle.optimizer.Adam(learning_rate=learning_rate, weight_decay=paddle.regularizer.L2Decay(0.0005), parameters=model.parameters())

    MAX_EPOCH = 2000
    for epoch in range(MAX_EPOCH):
        print(f"epoch {epoch}")
        model.train()
        for i, data in enumerate(train_loader()):
            img, gt_boxes, gt_labels, img_scale = data
            gt_scores = np.ones(gt_labels.shape).astype('float32')
            gt_scores = paddle.to_tensor(gt_scores)
            img = paddle.to_tensor(img)
            gt_boxes = paddle.to_tensor(gt_boxes)
            gt_labels = paddle.to_tensor(gt_labels)
            outputs = model(img)  #前向传播，输出[P0, P1, P2]
            loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                  anchors = ANCHORS,
                                  anchor_masks = ANCHOR_MASKS,
                                  ignore_thresh=IGNORE_THRESH,
                                  use_label_smooth=False)        # 计算损失函数

            loss.backward()    # 反向传播计算梯度
            opt.step()  # 更新参数
            opt.clear_grad()
            if i % 10 == 0:
                timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                print('{}[TRAIN]epoch {}, iter {}, output loss: {}'.format(timestring, epoch, i, loss.numpy()))

        # save params of model
        if (epoch % 5 == 0) or (epoch == MAX_EPOCH -1):
            paddle.save(model.state_dict(), 'yolo_epoch{}'.format(epoch))

        # 每个epoch结束之后在验证集上进行测试
        '''
        model.eval()
        for i, data in enumerate(valid_loader()):
            img, gt_boxes, gt_labels, img_scale = data
            gt_scores = np.ones(gt_labels.shape).astype('float32')
            gt_scores = paddle.to_tensor(gt_scores)
            img = paddle.to_tensor(img)
            gt_boxes = paddle.to_tensor(gt_boxes)
            gt_labels = paddle.to_tensor(gt_labels)
            outputs = model(img)
            loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                  anchors = ANCHORS,
                                  anchor_masks = ANCHOR_MASKS,
                                  ignore_thresh=IGNORE_THRESH,
                                  use_label_smooth=False)
            if i % 1 == 0:
                timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                print('{}[VALID]epoch {}, iter {}, output loss: {}'.format(timestring, epoch, i, loss.numpy()))
        '''
