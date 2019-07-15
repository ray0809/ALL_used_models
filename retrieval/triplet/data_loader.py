#coding=utf-8
import os
import sys
import keras
import random
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageOps
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.utils.data_utils import Sequence
from torchvision.transforms import *



class dataLoader(Sequence):
    def __init__(self, txt, batch_size, tag='train',transforms=None):
        self.data, self.labels = self._parsing(txt)
        self.tag = tag
        self.transforms = transforms
        self.label_set = set(self.labels)
        self.label2index = { i:list(np.where(np.array(self.labels)==i)[0]) for i in self.label_set}
        self.batch_size = batch_size
        if tag != 'train':
            '''
            依次遍历整个数据集，第i个数据，作为它的正pair，随机从所有该类别内选取一个，负样本随机选取一个异类，再随机选取其中一个样本。
            '''
            triplets = [[i, 
                        random.choice(self.label2index[self.labels[i]]), 
                        random.choice(self.label2index[random.choice(list(self.label_set - set([self.labels[i]])))])
                        ] 
                        for i in tqdm(range(len(self.data)))]
            

            
            self.test_triplet = triplets

    def __getitem__(self, idx):
        samples = range(idx * self.batch_size, (idx+1) * self.batch_size)
        # print('samples',samples)
        q_imgs, p_imgs, n_imgs = [], [], []

        for index in samples:
            if self.tag == 'train':  
                img1 = self._read(self.data[index])
                label = self.labels[index]
                
                postive_index = index
                while postive_index == index:
                    postive_index = random.choice(self.label2index[label])
                img2 = self._read(self.data[postive_index])
                
                negtive_label = random.choice(list(self.label_set - set([label])))
                negtive_index = random.choice(self.label2index[negtive_label])
                img3 = self._read(self.data[negtive_index])

                
        
            else:
                sample = self.test_triplet[index]
                # print(sample)
                img1 = self._read(self.data[sample[0]])
                img2 = self._read(self.data[sample[1]])
                img3 = self._read(self.data[sample[2]])
                # print(img1.shape, img2.shape, img3.shape)
        
            if self.transforms:
                img1 = self.transforms(img1)
                img2 = self.transforms(img2)
                img3 = self.transforms(img3)

            q_imgs.append(np.array(img1))
            p_imgs.append(np.array(img2))
            n_imgs.append(np.array(img3))

        return [np.array(q_imgs).astype('float32') / 255, np.array(p_imgs).astype('float32') / 255, np.array(n_imgs).astype('float32') / 255], None
    

    def _read(self, path):
        img = Image.open(path).convert('RGB')
        w, h = img.size
        if w != h:
            MAX = max(w, h)
            img = ImageOps.expand(img, border=(0, 0, MAX - w, MAX - h), fill=0)

        return img



    def _parsing(self, txt):
        paths = []
        labels = []
        with open(txt, 'r') as f:
            f = f.readlines()
            for i in f:
                path, label = i.strip().split(' ')
                paths.append(path)
                labels.append(int(label))
        return paths, labels


    def __len__(self):
        if self.tag == 'train':
            return len(self.data) // self.batch_size
        else:
            return len(self.test_triplet) // self.batch_size
    


if __name__ == '__main__':
    txt_path = sys.argv[1]
    testTrans = Compose([Resize(size=(224,224))])

    data = dataLoader(txt_path, 32, 'train', transforms=testTrans)

    data = iter(data)
    [i,j,k],m = next(data)
    print(i.shape, j.shape, k.shape)
    # a = GetSampling(main_path)
    # for i,j in data:
    # print(i.shape)