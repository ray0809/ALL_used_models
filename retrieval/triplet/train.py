#coding=utf-8
import os
import sys
import tensorflow as tf
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.losses import categorical_crossentropy

from utils import *
from data_loader import *
from model import *
from torchvision.transforms import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 


train_samples = './txt/train.txt'
test_samples = './txt/test.txt'


def main(save_path, date):

    ############################################################
    # 如果采用余弦距离的话，margin间隔建议设置在0.3左右
    # tripletnet,need_save_model = TripletNet(margin=0.5, dis='cos')
    tripletnet,need_save_model = TripletNet(margin=1, dis='euclidean')
    tripletnet.summary()

    adam = Adam(lr=1e-3, decay=0.7)
    sgd = SGD(lr=0.0001, momentum=0.9, decay=0.7)
    adroms = RMSprop(lr=0.001)
    tripletnet.compile(optimizer=adam, loss=None)

    

    trainTrans = Compose([Resize(size=(224,224))])
    train = dataLoader(train_samples, batch_size=24, tag='train', transforms=trainTrans)

    testTrans = Compose([Resize(size=(224,224))])
    test = dataLoader(test_samples, batch_size=24, tag='test', transforms=testTrans)
    modelcheck = MyModelCheckpoint('{}/{}.hdf5'.format(save_path, date),
                                    need_save_model,
                                    save_best_only=True,
                                    save_weights_only=False)



    # rdshechudle = ReduceLROnPlateau(factor=0.75,
    #                                 patience=3,
    #                                 min_lr=1e-6)
    rdshechudle = LearningRateScheduler(lambda epoch, lr: lr / (epoch + 1)**2)

    tripletnet.fit_generator(train,
            steps_per_epoch=train.__len__(),
            epochs=100,
            callbacks=[modelcheck, rdshechudle],
            validation_data=test,
            validation_steps=test.__len__(),
            use_multiprocessing=True,
            workers=8,
            max_queue_size=20
            )

if __name__ == '__main__':
    save_path = sys.argv[1]
    date = sys.argv[2]
    main(save_path, date)
