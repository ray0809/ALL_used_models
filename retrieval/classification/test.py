#coding=utf-8
import os
import sys
import shutil
import imghdr
import numpy as np
from time import time
from keras.layers import Input,Lambda
from keras.models import load_model,Model


from model import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def crtEmbedding(weights_path):
    model = crtModel(None, freezen=False, norm=True)
    model.load_weights(weights_path, by_name=True)
    return model

if __name__ == '__main__':
    mainpath = sys.argv[1]
    weights_path = sys.argv[2]
    test_path = sys.argv[3]
    types = int(sys.argv[4])  # 0 is retrieval and 1 is extract features
    fullpath = idx2path(mainpath)
    net = crtEmbedding(weights_path)

    if types:
        ExtractFeats(mainpath, net)
    else:
        feats = np.load('features/FullFeatures.npy')
        test_pics = os.listdir(test_path)
        for pic in test_pics:
            path = os.path.join(test_path, pic)
            if imghdr.what(path):
                begin = time()
                feat = ExtractOneFeat(path, net)
                end_extrc = time() - begin

                begin = time()
                result = np.dot(feat, feats.T)
                sort_result = np.argsort(result)[::-1][:21] # 取top20的结果
                end_dot = time() - begin

                print('extract time is %f\'s , dot time is %f\'s' % (end_extrc, end_dot))

                save_path = 'result/' + os.path.basename(path).split('.')[0]
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                for i,index in enumerate(sort_result):
                    p = fullpath[sort_result[i]]
                    shutil.copy(p, os.path.join(save_path,str(i)+'_'+os.path.basename(p)))