#coding=utf-8
import os
import warnings
import numpy as np
from tqdm import tqdm
from keras.callbacks import Callback
from keras.preprocessing import image



class MyModelCheckpoint(Callback):
    def __init__(self, filepath, mymodel, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MyModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.mymodel = mymodel

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            if os.path.isfile(filepath):
                                os.remove(filepath)
                            self.mymodel.save_weights(filepath, overwrite=True)
                        else:
                            if os.path.isfile(filepath):
                                os.remove(filepath)
                            self.mymodel.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                    self.mymodel.save_weights(filepath, overwrite=True)
                else:
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                    self.mymodel.save(filepath, overwrite=True)


def idx2path(mainpath):
    '''
    主要是为了图像跟特征的索引保持一致，后续测试检索返回结果图需要它
    '''
    fullpath = []
    for root, dirs, files in os.walk(mainpath):
        files = [f for f in files if not f[0] == '.'] # 屏蔽隐藏文件
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
    # img = image.load_img(path, target_size=target_size, interpolation='bilinear')
    img = image.load_img(path, grayscale=False)
    img = np.expand_dims(image.img_to_array(img), 0)
    feat = net.predict(img.astype('float32'))
    return feat[0]
