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





def cal_map(labels, feats, topK):
    print('>>>开始计算test数据集的mAP')
    average_precision = 0.0
    for i in tqdm(range(feats.shape[0])):
        feat = feats[i]
        result = np.sum(np.square(feats - feat), axis=-1)
        # print('result',result.shape)
        sort_indx = np.argsort(result)[1:topK+1] # 取top20的结果

        same_label_indx = (labels[sort_indx] == labels[i])

        if same_label_indx.sum() == 0:
            continue

        average_precision += (np.cumsum(same_label_indx) / np.linspace(1, topK, topK)).sum() / same_label_indx.sum()

    mean_average_precision = average_precision / feats.shape[0]
    return mean_average_precision



