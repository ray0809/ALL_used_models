#coding=utf-8
import os
import sys
import cv2
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

from keras.callbacks import *
from albumentations import *

from data_loader import *
from utils import *


BACKBONE = 'resnet18'
CLASSNB = 2
HEIGHT = 256
WIDTH = 256
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main(imgfile, maskfile):
    train_aug = Compose([Resize(height=HEIGHT,width=WIDTH,interpolation=cv2.INTER_AREA),
                RandomRotate90(p=0.5),
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                HueSaturationValue(p=0.5),
                RandomBrightnessContrast(0.5),
                Transpose(p=0.5)])

    val_aug = Compose([Resize(height=HEIGHT,width=WIDTH,interpolation=cv2.INTER_AREA)])

    # define model
    model = Unet(BACKBONE, encoder_weights='imagenet',
                input_shape=(HEIGHT, WIDTH, 3),
                classes=CLASSNB,
                encoder_freeze=False,
                activation='softmax')
    model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
    model.load_weights('weights/20190220_res18.h5')

    # define dataloader
    train_samples, val_samples = GetSampling(imgfile, maskfile)
    trainloader = DataLoader(train_samples, CLASSNB, train_aug, 16)
    valloader = DataLoader(val_samples, CLASSNB, val_aug, 16)


    modelcheck = FixModelCheckpoint('weights/20190220.h5',
                                save_best_only=True,
                                save_weights_only=True)
    rscheduler = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
    model.fit_generator(trainloader,
                        steps_per_epoch=trainloader.__len__(),
                        epochs=100,
                        verbose=1,
                        callbacks=[modelcheck, rscheduler],
                        validation_data=valloader,
                        validation_steps=valloader.__len__(),
                        max_queue_size=10,
                        workers=20,
                        use_multiprocessing=True,
                        shuffle=True)

if __name__ == '__main__':
    imgfile, maskfile = sys.argv[1], sys.argv[2]
    main(imgfile, maskfile)