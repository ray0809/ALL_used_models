#coding=utf-8
import os
import sys
from keras.optimizers import *
from keras.callbacks import *
from keras.losses import categorical_crossentropy

from utils import *
from data_loader import *
from model import *
from torchvision.transforms import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(main_path, save_path, date):

    # 如果采用余弦距离的话，margin间隔建议设置在0.5左右
    tripletnet,need_save_model = TripletNet(margin=0.6, dis='cos')
    tripletnet.summary()
    tripletnet.load_weights(os.path.join(save_path,'20190218.hdf5'))
    adam = Adam(lr=1e-4)
    tripletnet.compile(optimizer=adam,loss=lambda y_true,y_pred: y_pred)

    train_samples, test_samples = GetSampling(main_path)

    trainTrans = Compose([RandomRotation(20,expand=True),
							RandomHorizontalFlip(0.5),
							Resize(size=(224,224))])
    train = dataLoader(train_samples, batch_size=5, transformer=trainTrans, shuffle=True)

    testTrans = Compose([Resize(size=(224,224))])
    test = dataLoader(test_samples, batch_size=5, transformer=testTrans, shuffle=True)
    modelcheck = MyModelCheckpoint('{}/{}.hdf5'.format(save_path,date),
                                    need_save_model,
                                    save_best_only=True,
                                    save_weights_only=False)
    lrscheduler = ReduceLROnPlateau(factor=0.5, patience=1, min_lr=1e-6)
    tripletnet.fit_generator(train,
            steps_per_epoch=train.__len__(),
            epochs=100,
            callbacks=[modelcheck,lrscheduler],
            validation_data=test,
            validation_steps=test.__len__(),
            use_multiprocessing=True,
            workers=20,
            max_queue_size=20
            )

if __name__ == '__main__':
    main_path = sys.argv[1]
    save_path = sys.argv[2]
    date = sys.argv[3]
    main(main_path, save_path, date)
