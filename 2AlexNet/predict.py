# -*- coding:utf-8 -*-

from alexnet import *
import os
from keras.preprocessing.image import load_img, img_to_array
import csv

alexnet = AlexNet(227, 227)
model = alexnet.alexnet()
model.load_weights('first_try.h5')

test_dir = '/data1/kaggle/dog_cat/test1'
pic_list = []
for num in range(1, 12501):
    pic_list.append(os.path.join(test_dir, str(num) + '.jpg'))

imgdatas = np.ndarray((12500, 227, 227, 3), dtype=np.uint8)
i = 0
for pic in pic_list:
    img = load_img(pic, grayscale=False, target_size=[227, 227])
    img = img_to_array(img)
    imgdatas[i] = img / 255
    i += 1

result = model.predict(imgdatas, batch_size=1, verbose=1)
# print(result)
# print(np.argmax(result,axis=1))

# 0:cat, 1:dog
writelist = []
for i in range(0, 12500):
    writelist.append([i+1] + [list(np.argmax(result, axis=1))[i]])
with open("/data1/kaggle/dog_cat/alexnet.csv","w") as csvfile:
    writer = csv.writer(csvfile)

    #先写入columns_name
    writer.writerow(["id", "label"])
    #写入多行用writerows
    writer.writerows(writelist)