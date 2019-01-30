#coding=utf-8
import os
import sys
import random


def crtTxt(path):
    all_path = []
    label_count = 0
    for root, dirs, files in os.walk(path):
        if len(files) > 60:
            for f in files:
                if 'jpg' in f:
                    all_path.append([os.path.join(root,f), str(label_count)])
            label_count += 1
    random.shuffle(all_path)
    train = all_path[:int(len(all_path)*0.9)]
    test = all_path[int(len(all_path)*0.9):]
    with open('train.txt','w') as f:
        for i in train:
            f.write(i[0] + ' ' + i[1] + '\n')
    
    with open('test.txt','w') as f:
        for i in test:
            f.write(i[0] + ' ' + i[1] + '\n')


if __name__ == '__main__':
    path = sys.argv[1]
    crtTxt(path)