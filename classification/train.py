#coding=utf-8
import os
import sys
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy

from data_loader import *
from model import *
from torchvision.transforms import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(main_path, train_txt, test_txt, class_nb):

    ############################################################
    '''
    当搭建的模型是由多个其他的base model拼凑而成的，那么直接加载之前冻层训练的权重会出错
    需要先在冻结前加载，随后遍历将所有解冻
    '''
    resnet = crtModel(class_nb, freezen=True)
    # resnet.load_weights('weights/20190130_best.h5')
    # for layer in resnet.layers:
    #     if isinstance(layer, Model):
    #         for ly in layer.layers:
    #             ly.trainable = True
    #     else:
    #         layer.trainable = True
    ############################################################
    resnet.summary()
    adam = Adam(lr=0.001)
    resnet.compile(optimizer=adam,loss=categorical_crossentropy,metrics=['accuracy'])

    trainTrans = Compose([RandomRotation(20,expand=True),
							RandomHorizontalFlip(0.5),
							Resize(size=(224,224))])
    train = dataLoader(class_nb, main_path, train_txt, batch_size=16, transformer=trainTrans, shuffle=True)

    testTrans = Compose([Resize(size=(224,224))])
    test = dataLoader(class_nb, main_path, test_txt, batch_size=16, transformer=testTrans)
    modelcheck = ModelCheckpoint('weights/20190130_best.h5',save_best_only=True,save_weights_only=False)
    resnet.fit_generator(train,
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
    train_txt = sys.argv[2]
    test_txt = sys.argv[3]
    class_nb = int(sys.argv[4])
    main(main_path, train_txt, test_txt, class_nb)
