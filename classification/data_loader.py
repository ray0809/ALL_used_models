#coding=utf-8
import os
import sys
import keras
import random
import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.utils.data_utils import Sequence
from torchvision.transforms import *


class dataLoader(Sequence):
    def __init__(self, class_nb, main_path, txt_path, batch_size, shuffle=False, transformer=None):
        with open(txt_path, 'r') as f:
            self.samples = f.readlines()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transformer = transformer
        self.main_path = main_path
        self.class_nb = class_nb
        
    def __getitem__(self, idx):
        samples = self.samples[idx*self.batch_size:(idx+1)*self.batch_size]
        # print('samples',samples)
        imgs, labels = [], []
        for sample in samples:
            one_line = sample.strip().split()
            img_path = os.path.join(self.main_path,one_line[0])
            label = int(one_line[1])
            try:
                img = image.load_img(img_path)
                if self.transformer is not None:
                    img = self.transformer(img)
                imgs.append(image.img_to_array(img))
                labels.append(label)
            except:
                print('wrong img_path is:',img_path)
                continue
        x, y = np.array(imgs), to_categorical(np.array(labels), num_classes=self.class_nb)
        # print(x.shape,y.shape)
        return x, y 

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples) // self.batch_size

if __name__ == '__main__':
    main_path = sys.argv[1]
    train_txt = sys.argv[2]
    test_txt = sys.argv[3]
    class_nb = int(sys.argv[4])
    testTrans = Compose([Resize(size=(224,224))])
    data = dataLoader(class_nb,main_path,train_txt,32,transformer=testTrans)

    data = iter(data)
    i,j = next(data)
    # for i,j in data:
    print(i.shape,j.shape)