#coding=utf-8
import os
import warnings
import numpy as np
import keras.backend as K
from keras.callbacks import Callback


class FixModelCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(FixModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

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
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            if os.path.isfile(filepath):
                                os.remove(filepath)
                            self.model.save(filepath, overwrite=True)
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
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                    self.model.save(filepath, overwrite=True)




class EvalIOU(Callback):
    def __init__(self, filepath, Eval, generator,
                mymodel,
                factor=0.25,
                patience=1,
                init_lr=1e-3,
                min_lr=1e-6):
        super(EvalIOU, self).__init__()
        self.filepath = filepath
        self.Eval = Eval
        self.class_nb = generator.class_nb
        self.len = len(generator)
        self.generator = generator
        self.mymodel = mymodel
        self.best = -np.Inf
        self.patience = patience
        self.wait = 0
        self.factor = factor
        self.init_lr = init_lr
        self.min_lr = min_lr

    def _reset(self):
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        eval = self.Eval(self.class_nb)
        for i, (imgs, masks) in enumerate(self.generator):
            if i < self.len:
                predict = self.model.predict_on_batch(imgs)
                eval.addBatch(predict, masks)
            else:
                break
        overall_acc, per_class_acc, per_class_iu, mIOU = eval.getMetric()
        if mIOU > self.best:
            self.wait = 0
            self.best = mIOU
            if os.path.isfile(self.filepath):
                os.remove(self.filepath)
            self.mymodel.save_weights(self.filepath, overwrite=True)
        else:
            self.wait += 1
            if self.wait > self.patience:
                new_lr = self.init_lr * (1 - self.factor)
                new_lr = np.clip(new_lr, self.min_lr, self.init_lr)
                K.set_value(self.model.optimizer.lr, new_lr)
                self.wait = 0
        print('current lr is {}'.format(K.get_value(self.model.optimizer.lr)))
        print('overall_acc is {}'.format(overall_acc))
        print('per_class_acc is {}'.format(per_class_acc))
        print('per_class_iu is {}'.format(per_class_iu))
        print('mIOU is {} (except background)'.format(mIOU))
        

def poly_lr_scheduler(epoch, lr, max_epochs=100, power=0.9):
    lr = round(lr * (1 - epoch / max_epochs) ** power, 8)
    return lr