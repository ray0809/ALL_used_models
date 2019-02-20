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


def WriteTxt(samples, mode):
    if not os.path.isdir('./txt'):
        os.makedirs('./txt')
    with open('./txt/{}.txt'.format(mode),'w') as f:
        for sample in samples:
            for i in sample:
                f.write(i + '\n')


def GetSampling(main_path):
    samples = []
    for root, dirs, files in os.walk(main_path):
        if len(files) > 90:
            random.shuffle(files)
            samples.append([os.path.join(root,f) for idx,f in enumerate(files) if 'jpg' in f])
    train_samples = [i[:int(len(i)*0.9)] for i in samples]
    test_samples = [i[int(len(i)*0.9):] for i in samples]
    WriteTxt(train_samples,'train')
    WriteTxt(test_samples,'test')
    return train_samples, test_samples

def TripeltSampling(samples):
    full_samples = []
    min_nb = min([len(d) for d in samples]) - 1  # 最小类别数
    for idx, sample in enumerate(samples):
        random.shuffle(sample)
        for i in range(min_nb):
            q, p = sample[i], sample[i + 1]
            inc = random.randrange(1, len(samples))
            dn = (int(idx) + inc) % len(samples)
            n = samples[dn][i]
            full_samples.append([q, p, n])
    random.shuffle(full_samples)
    # print('full_samples',len(full_samples))
    return full_samples

class dataLoader(Sequence):
    def __init__(self, samples, batch_size, shuffle=True, transformer=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transformer = transformer
        self.samples = samples
        self.tripsamples = TripeltSampling(samples)

    def __getitem__(self, idx):
        samples = self.tripsamples[idx*self.batch_size:(idx+1)*self.batch_size]
        # print('samples',samples)
        q_imgs, p_imgs, n_imgs = [], [], []
        none_label = []
        for one_line in samples:
            q_path = one_line[0]
            p_path = one_line[1]
            n_path = one_line[2]
            try:
                q_img = self.read_transformer(q_path)
                p_img = self.read_transformer(p_path)
                n_img = self.read_transformer(n_path)
                q_imgs.append(image.img_to_array(q_img))
                p_imgs.append(image.img_to_array(p_img))
                n_imgs.append(image.img_to_array(n_img))
                none_label.append(1)
            except:
                continue
        # print(x.shape,y.shape)
        return [np.array(q_imgs), np.array(p_imgs), np.array(n_imgs)], np.array(none_label)

    def on_epoch_end(self):
        if self.shuffle:
            self.tripsamples = TripeltSampling(self.samples)

    def __len__(self):
        return len(self.tripsamples) // self.batch_size

    def read_transformer(self, path):
        img = image.load_img(path)
        if self.transformer:
            img = self.transformer(img)
        return img

    


if __name__ == '__main__':
    main_path = sys.argv[1]
    testTrans = Compose([Resize(size=(224,224))])
    trisam = TripeltSampling(GetSampling(main_path))
    data = dataLoader(trisam,1,transformer=testTrans)

    data = iter(data)
    i,j = next(data)
    for a in i:
        print(a.shape)
    # a = GetSampling(main_path)
    # for i,j in data:
    # print(i.shape)