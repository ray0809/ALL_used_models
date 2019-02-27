#coding=utf-8
import os
import sys
import random
import cv2
import numpy as np
from keras.utils import Sequence
from keras.utils import to_categorical
from albumentations import *

# 制作数据txt的时候，滤除掉隐藏文件 .开头的
def DelHide(files):
    return [f for f in files if not f[0] == '.']

# 随机采样train和val后，对它们进行保存，以备后续测试可视化
def WriteTxt(samples, mode):
    if not os.path.isdir('./txt'):
        os.makedirs('./txt')
    with open('./txt/{}.txt'.format(mode),'w') as f:
        for img, mask in samples:
            f.write(img + ' ' + mask + '\n')

def GetSampling(imgFile, maskFile):
    # imgs = DelHide(os.listdir(imgFile))
    masks = DelHide(os.listdir(maskFile))
    # imgs = list(sorted(imgs))
    masks = list(sorted(masks))  # 并行抽取图像和mask的文件
    # assert len(imgs) == len(masks)
    random_nb = random.sample(range(len(masks)), int(len(masks)*0.9))
    train_samples = []
    val_samples = []
    for i in range(len(masks)):
        imgPath = os.path.join(imgFile,masks[i].replace('png','jpg'))
        maskPath = os.path.join(maskFile, masks[i])
        if os.path.isfile(imgPath):
            if i in random_nb:
                train_samples.append([imgPath, maskPath])
            else:
                val_samples.append([imgPath, maskPath])
    WriteTxt(train_samples, 'train')
    WriteTxt(val_samples, 'val')
    return train_samples, val_samples



class DataLoader(Sequence):
    def __init__(self, samples, 
                    class_nb, 
                    transformer, 
                    batch_size=16,
                    activation='softmax', 
                    shuffle=True):
        self.samples = samples
        self.batch_size = batch_size
        self.class_nb = class_nb
        self.transformer = transformer
        self.shuffle = shuffle
        self.activation = activation
    
    def __getitem__(self, idx):
        batch_samples = self.samples[idx*self.batch_size:(idx+1)*self.batch_size]
        imgs, masks = [], []
        for imgPath, maskPath in batch_samples:
            
            img = cv2.imread(imgPath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(maskPath, 0)

            img = self._padding(img)
            mask = self._padding(mask)

            # print('1',img.shape, mask.shape)
            augmented = self.transformer(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            # print('2',img.shape, mask.shape)

            
            if self.activation == 'softmax':
                # 如果是用多分类交叉熵，则需要one-hot编码
                new_h, new_w = mask.shape
                mask = to_categorical(mask.reshape(-1), self.class_nb)
                mask = mask.reshape(new_h, new_w, self.class_nb)
            else:
                #二分类，使用sigmoid，则不需要
                mask = np.expand_dims(mask, axis=-1)
            imgs.append(img)
            masks.append(mask)

        imgs = np.array(imgs) / 127.0 - 1
        masks = np.array(masks)
        return imgs, masks 

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples) // self.batch_size

    def _padding(self, img):
        shape = img.shape
        h, w = shape[:2]
        width = np.max([h, w])
        padd_h = (width - h) // 2
        padd_w = (width - w) // 2
        if len(shape) == 3:
            padd_tuple = ((padd_h,width-h-padd_h),(padd_w,width-w-padd_w),(0,0))
        else:
            padd_tuple = ((padd_h,width-h-padd_h),(padd_w,width-w-padd_w))
        img = np.pad(img, padd_tuple, 'constant')
        return img    



if __name__ == '__main__':
    imgfile = sys.argv[1]
    maskfile = sys.argv[2]
    a, b = GetSampling(imgfile, maskfile)
    aug = Compose([Resize(height=128,width=128,interpolation=cv2.INTER_AREA),
               RandomRotate90(p=0.5),
              VerticalFlip(p=0.5),
              HorizontalFlip(p=0.5),
              ChannelShuffle(p=0.5),
              HueSaturationValue(p=0.5),
              RandomBrightnessContrast(0.5),
              Transpose(p=0.5)])
    data = DataLoader(a, 3, aug, 16)
    for i, (imgs, masks) in enumerate(data):
        print('这是第%d次迭代。。。' % i)
        print('imgs shape:', imgs.shape, 'mask shape:', masks.shape)