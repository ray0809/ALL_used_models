#coding=utf-8
import os
import sys
import keras
import random
import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.utils.data_utils import Sequence
from torchvision.transforms import *


class dataLoader(Sequence):
    def __init__(self, txt_path, batch_size, shuffle=False):
        with open(txt_path, 'r') as f:
            self.samples = f.readlines()
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __getitem__(self, idx):
        samples = self.samples[idx*self.batch_size:(idx+1)*self.batch_size]
        imgs, multi_labels = [], []
        for sample in samples:
            one_line = sample.strip().split()
            img_path = one_line[0]
            labels = [int(i) for i in one_line[1:]]
            try:
                img_path = os.path.join('/home/datalab/ex_disk2/hanbing/data/multi_attrs/nuswide/Flickr',img_path)
                img = image.load_img(img_path,target_size=(224,224))
                imgs.append(image.img_to_array(img))
                multi_labels.append(labels)
            except:
                print('wrong img_path is:',img_path)
                continue
        return np.array(imgs), np.array(multi_labels)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples) // self.batch_size

if __name__ == '__main__':
    train_txt = sys.argv[1]
    print(train_txt)
    # test_txt = sys.argv[2]

    data = dataLoader(train_txt,32)
    data = iter(data)
    i,j = next(data)
    print(i.shape,j.shape)