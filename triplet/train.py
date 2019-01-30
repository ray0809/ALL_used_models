#coding=utf-8
import os
import sys
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy

from utils import *
from data_loader import *
from model import *
from torchvision.transforms import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(main_path):

    ############################################################
    '''
    当搭建的模型是由多个其他的base model拼凑而成的，那么直接加载之前冻层训练的权重会出错
    需要先在冻结前加载，随后遍历将所有解冻
    '''
    tripletnet,need_save_model = TripletNet(freezen=False,margin=1)
    # for layer in resnet.layers:
    #     if isinstance(layer, Model):
    #         for ly in layer.layers:
    #             ly.trainable = True
    #     else:
    #         layer.trainable = True
    ############################################################
    tripletnet.summary()
    tripletnet.compile(optimizer='adam',loss=lambda y_true,y_pred: y_pred)

    samples = GetSampling(main_path)
    train_samples = [i[:int(len(i)*0.9)] for i in samples]
    test_samples = [i[int(len(i)*0.9):] for i in samples]

    trainTrans = Compose([RandomRotation(20,expand=True),
							RandomHorizontalFlip(0.5),
							Resize(size=(224,224))])
    train = dataLoader(train_samples, batch_size=8, transformer=trainTrans, shuffle=True)

    testTrans = Compose([Resize(size=(224,224))])
    test = dataLoader(test_samples, batch_size=8, transformer=testTrans)
    modelcheck = MyModelCheckpoint('weights/20190130_best.h5',need_save_model,save_best_only=True,save_weights_only=False)
    tripletnet.fit_generator(train,
            steps_per_epoch=train.__len__(),
            epochs=10,
            callbacks=[modelcheck],
            validation_data=test,
            validation_steps=test.__len__(),
            use_multiprocessing=True,
            workers=20,
            max_queue_size=20
            )

if __name__ == '__main__':
    main_path = sys.argv[1]
    main(main_path)
