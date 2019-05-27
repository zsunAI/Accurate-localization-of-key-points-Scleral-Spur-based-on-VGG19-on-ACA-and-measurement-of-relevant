import tensorflow as tf
import numpy as np
import cv2
import glob
import random
import  os

pathes = glob.glob('data1/*.jpg')
img_crop_total = []
label_crop_total = []
labels = []
with open('location1.txt') as f:
    for x in f.readlines():
        labels.append([int(i) for i in x.strip().split(' ')])

for idx, path in enumerate(pathes):
    iidx = int(path.split('\\')[1].split('.')[0]) - 1
    img = cv2.imread(path)
    if img.shape[0] == 280:
        for i in range(5):
            # 进行中心的crop
            if i != 0:
                x = img.shape[0]
                y = img.shape[1]
                random_scale_x = random.randrange(0, int(x - 224))
                random_scale_y = random.randrange(0, int(y - 224))
                img_crop = img[random_scale_y:random_scale_y + 224, random_scale_x:random_scale_x + 224, :]
                label = [labels[iidx][0] - random_scale_x, labels[iidx][1] - random_scale_y]
                img_crop_total.append(img_crop)
                label_crop_total.append(label)
            else:
                x = img.shape[0]
                y = img.shape[1]
                scale_x = int((x - 224) / 2)
                scale_y = int((y - 224) / 2)
                img_crop = img[scale_y:scale_y + 224, scale_x:scale_x + 224]
                img_crop_total.append(img_crop)
                label = [labels[iidx][0] - scale_x, labels[iidx][1] - scale_y]
                label_crop_total.append(label)
    else:
        img_crop_total.append(img)
        label_crop_total.append(labels[idx])

# 进行洗牌操作
index = np.arange(0, len(img_crop_total))
random.shuffle(index)
x = np.array(img_crop_total)
y = np.array(label_crop_total)
x_shuffle = x[index]
y_shuffle = y[index]
# print(np.shape(img_crop_total))
# # print(np.shape(label_crop_total))

print(x_shuffle[0:3].shape)
# 拆分成训练集和测试集数据
p = int(len(x) * 0.85)
train_x = x_shuffle[:p]
train_y = y_shuffle[:p]
test_x = x_shuffle[p:]
test_y = y_shuffle[p:]


if not os.path.exists('npy'):
    os.makedirs('npy')
np.save('npy/x_train', train_x)
np.save('npy/y_train', train_y)
np.save('npy/x_test', test_x)
np.save('npy/y_test', test_y)

# for i in range(10):
#     img = img_crop_total[i]
#     clone_img_1 = img.copy()
#     print(img.shape)
#     cv2.circle(clone_img_1, (label_crop_total[i][0], label_crop_total[i][1]), 3, (0, 0, 255), -1)
#     cv2.imshow('img', clone_img_1)
#     cv2.waitKey(0)





