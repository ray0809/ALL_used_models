#coding=utf-8
import os
import sys
import random
import numpy as np
from tqdm import tqdm
from keras.preprocessing import image



def crtTxt(path):
    '''
    做txt主要目的是为了知道哪个类别对应的数字，如果不在意的话，其实这一步可以放到datalaoder里面
    '''
    all_path = []
    label_count = 0
    for root, dirs, files in os.walk(path):
        if len(files) > 60: # # 我的有些类别数量太少，我直接剔除少于60的
            for f in files:
                if 'jpg' in f: # 我只有jpg
                    all_path.append([os.path.join(root,f), str(label_count)])
            label_count += 1 # 处理完一个类就加一
    random.shuffle(all_path)
    train = all_path[:int(len(all_path)*0.9)]
    test = all_path[int(len(all_path)*0.9):]
    if not os.path.isdir('txt/'):
        os.makedirs('txt/')
    with open('txt/train.txt','w') as f:
        for i in train:
            f.write(i[0] + ' ' + i[1] + '\n')
    
    with open('txt/test.txt','w') as f:
        for i in test:
            f.write(i[0] + ' ' + i[1] + '\n')


def idx2path(mainpath):
    '''
    主要是为了图像跟特征的索引保持一致，后续测试检索返回结果图需要它
    '''
    fullpath = []
    for root, dirs, files in os.walk(mainpath):
            for f in files:
                if 'jpg' in f:
                    fullpath.append(os.path.join(root,f))
    fullpath = np.sort(fullpath)
    return fullpath


def ExtractFeats(mainpath, net, full=True):
    path = idx2path(mainpath)
    result = []
    for i, p in enumerate(tqdm(path)):
        feat = ExtractOneFeat(p, net)
        result.append(feat)
    result = np.matrix(result)
    if not os.path.isdir('features'):
        os.makedirs('features/')
    np.save('features/FullFeatures.npy',result)
    return 

def ExtractOneFeat(path, net, target_size=(224,224)):
    img = image.load_img(path, target_size=target_size, interpolation='bilinear')
    img = np.expand_dims(image.img_to_array(img), 0)
    feat = net.predict(img.astype('float32'))
    return feat[0]




if __name__ == '__main__':
    path = sys.argv[1]
    crtTxt(path)
