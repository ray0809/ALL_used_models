#coding=utf-8
import os
import random
import joblib
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from keras.models import load_model

from model import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def LoadModel(weights_path):
    model = load_model(weights_path)
    return model

if __name__ == '__main__':
    weights_path = sys.argv[1]
    results_path = sys.argv[2]
    test_path = sys.argv[3]
    net = LoadModel(weights_path)
    

    

    if os.path.isfile(results_path):
        paths, labels, feats = joblib.load(results_path)
    
    else:
        print('>>>第一次运行，开始生成test数据集的特征向量')
        paths = []
        labels = []
        feats = []

        with open(test_path, 'r') as f:
            for i in tqdm(f.readlines()):
                path, label = i.strip().split()
                img = Image.open(path).convert('RGB').resize((224,224))
                feat = net.predict(np.expand_dims(np.array(img).astype('float32') / 255, 0))[0]
                paths.append(path)
                labels.append(int(label))
                feats.append(feat)

        labels = np.array(labels)
        feats = np.array(feats)
        joblib.dump([paths, labels, feats], results_path)


    mAP = cal_map(labels, feats, topK=10)
    print('>>>计算得到的mAP：{:.4f}'.format(mAP))


    print('\n')
    print('>>>随机挑选某个测试数据进行结果展示')

    nb = random.randint(0, len(paths) - 1)
    feat = feats[nb]
    result = np.sum(np.square(feats - feat), axis=-1)
    sort_result = np.argsort(result)[1:21] # 取top20的结果

    save_path = './results/{}/'.format(nb)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        shutil.copy(paths[nb], os.path.join(save_path, 'query_' + os.path.basename(paths[nb])))

    for i, indx in enumerate(sort_result):
        p = paths[indx]
        shutil.copy(p, os.path.join(save_path, 'result_' + str(i) + '_' + os.path.basename(p)))
    print('>>>检索结果已经写入results文件夹内')