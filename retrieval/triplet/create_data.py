#coding=utf-8
import os
import  scipy.misc as im
from keras.datasets import *



DATANAME = 'cifar10'
train, test = cifar10.load_data()

test_imgs = train[0]
test_label = train[1]

for i in range(test_imgs.shape[0]):
    img = test_imgs[i]
    label = str(test_label[i])
    path = '/home/datalab/ex_disk2/hanbing/data/classification/cifar10/'+label
    if not os.path.isdir(path):
        os.makedirs(path)
    im.imsave(os.path.join(path,label+'_'+str(i)+'.jpg'),img)