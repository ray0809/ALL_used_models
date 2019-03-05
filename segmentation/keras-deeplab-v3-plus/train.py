#coding=utf-8
import os
import sys
import cv2
import warnings

from albumentations import *
from segmentation_models.losses import bce_jaccard_loss,bce_dice_loss
from segmentation_models.metrics import iou_score

from keras.losses import categorical_crossentropy
from keras.utils import multi_gpu_model
from keras.callbacks import *


from model import *
from data_loader import *
from utils import *
from data_info import *
from eval_iou import *

warnings.filterwarnings("ignore")

CLASSNB = 1 # 如果是用二类交叉熵的话，使用sigomid，最后只需要1，不需要one-hot编码
HEIGHT = 512
WIDTH = 512
ACTIVATION = 'sigmoid'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"





def main(imgfile, maskfile):
    # define data aug
    train_aug = Compose([Resize(height=HEIGHT,width=WIDTH,interpolation=cv2.INTER_CUBIC),
                RandomRotate90(p=0.5),
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                HueSaturationValue(p=0.5),
                RandomBrightnessContrast(0.5),
                Transpose(p=0.5)])
    val_aug = Compose([Resize(height=HEIGHT,width=WIDTH,interpolation=cv2.INTER_CUBIC)])

    # define model
    model = crtModel(HEIGHT, WIDTH, CLASSNB, 
                    activation=ACTIVATION,
                    backbone='mobilenetv2')
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile('Adam', loss=bce_dice_loss, metrics=[iou_score])
    
    # define dataloader
    train_samples, val_samples = GetSampling(imgfile, maskfile)
    trainloader = DataLoader(train_samples, CLASSNB, train_aug, 4)
    valloader = DataLoader(val_samples, CLASSNB, val_aug, 4)

    # define data info
    data_infos = LoadData(train_samples, CLASSNB)
    data_infos.readFiles()
    print('类别权重：',data_infos.classWeights)

    # define metric
    path = '/home/datalab/ex_disk2/hanbing/weights/segmentation/deeplabv3+/'
    evaliou = EvalIOU(path+'/20190226_hair_face_crossentropy.h5',
                        Eval=Eval, 
                        generator=valloader,
                        mymodel=model)

    # define traing
    parallel_model.fit_generator(trainloader,
                        steps_per_epoch=len(trainloader),
                        epochs=20,
                        verbose=1,
                        callbacks=[evaliou],
                        class_weight=data_infos.classWeights,
                        max_queue_size=10,
                        workers=10,
                        use_multiprocessing=True,
                        shuffle=True)

if __name__ == '__main__':
    imgfile, maskfile = sys.argv[1], sys.argv[2]
    main(imgfile, maskfile)