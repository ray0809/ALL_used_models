#coding=utf-8
import os
import sys
import cv2
import warnings

from segmentation_models import *
from segmentation_models.losses import *
from segmentation_models.metrics import *

from keras.optimizers import *
from keras.callbacks import *
from albumentations import *

from data_loader import *
from utils import *
from data_info import *
from eval_iou import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings("ignore")


BACKBONE = 'resnet18'
CLASSNB = 1 # 如果是用二类交叉熵的话，使用sigomid，最后只需要1，不需要one-hot编码
HEIGHT = 256
WIDTH = 256
ACTIVATION='sigmoid'







def main(imgfile, maskfile):
    train_aug = Compose([Resize(height=HEIGHT,width=WIDTH,interpolation=cv2.INTER_CUBIC),
                RandomRotate90(p=0.5),
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                HueSaturationValue(p=0.5),
                RandomBrightnessContrast(0.5),
                Transpose(p=0.5)])

    val_aug = Compose([Resize(height=HEIGHT,width=WIDTH,interpolation=cv2.INTER_CUBIC)])


    # define unet model
    # model = Unet(BACKBONE, encoder_weights='imagenet',
    #             input_shape=(None, None, 3),
    #             classes=CLASSNB,
    #             encoder_freeze=False,
    #             activation='softmax')

    # define pspnet model
    # model = PSPNet(backbone_name=BACKBONE,
    #             input_shape=(HEIGHT, WIDTH, 3),
    #             classes=1,
    #             activation='sigmoid',
    #             encoder_weights='imagenet',
    #             psp_dropout=0.5
    #             )

    # define fpn model
    model = FPN(backbone_name=BACKBONE,
                input_shape=(HEIGHT, WIDTH, 3),
                classes=CLASSNB,
                activation=ACTIVATION,
                encoder_weights='imagenet',
                pyramid_dropout=0.5)

    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss=bce_jaccard_loss, metrics=[iou_score])
    model.summary()

    # define dataloader
    train_samples, val_samples = GetSampling(imgfile, maskfile)

    # define data info
    # data_infos = LoadData(train_samples, CLASSNB)
    # data_infos.readFiles()
    
    trainloader = DataLoader(samples=train_samples, 
                                class_nb=CLASSNB, 
                                transformer=train_aug, 
                                batch_size=16,
                                activation=ACTIVATION,
                                shuffle=True)
    valloader = DataLoader(samples=val_samples, 
                                class_nb=CLASSNB, 
                                transformer=val_aug, 
                                batch_size=16,
                                activation=ACTIVATION,
                                shuffle=False)

    path = '/home/datalab/ex_disk2/hanbing/weights/segmentation/skin/'
    model.load_weights(path+'20190227_res18_fpn_bce_jaccard_256.h5')
    modelcheck = FixModelCheckpoint(filepath=path+'/20190227_res18_fpn_bce_jaccard.h5',
                                    save_best_only=True,
                                    save_weights_only=True)
    rdlr = ReduceLROnPlateau(factor=0.75,
                            patience=2,
                            min_lr=1e-6)
    # evaliou = EvalIOU(filepath=path+'/20190227_res18_fpn_bce_jaccard.h5',
    #                     Eval=Eval, 
    #                     generator=valloader,
    #                     factor=0.25,
    #                     patience=2,
    #                     init_lr=1e-3,
    #                     min_lr=1e-6)
    model.fit_generator(trainloader,
                        steps_per_epoch=len(trainloader),
                        epochs=100,
                        verbose=1,
                        callbacks=[modelcheck, rdlr],
                        validation_data=valloader,
                        validation_steps=len(valloader),
                        max_queue_size=10,
                        workers=10,
                        use_multiprocessing=True,
                        shuffle=True)

if __name__ == '__main__':
    # python train.py /home/datalab/ex_disk2/hanbing/data/segmentation/facecutimg/srcimg/ /home/datalab/ex_disk2/hanbing/data/segmentation/face_hair/maskimg/
    imgfile, maskfile = sys.argv[1], sys.argv[2]
    main(imgfile, maskfile)