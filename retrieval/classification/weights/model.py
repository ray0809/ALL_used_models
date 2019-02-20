#coding=utf-8
import keras.backend as K
from keras.models import Model
from keras.layers import Input,Flatten,Dropout,Dense,Lambda
from keras.applications import resnet50

def Normalize(x):
    x = K.l2_normalize(x, axis=-1)
    return x

def crtModel(class_nb, freezen=True, norm=False):
    base_model = resnet50.ResNet50(include_top=False,pooling='avg')
    if freezen:
        for layer in base_model.layers:
            layer.trainable = False
    inputs = Input(shape=(None,None,3))
    x = Lambda(resnet50.preprocess_input,arguments={'mode':'tf'})(inputs)
    x = base_model(x)
    # x = Flatten()(x)
    # x = Dropout(0.5)(x)
    x = Dense(512,name='embedding')(x)
    if norm:
        out_embedding = Lambda(Normalize,name='l2_normalization')(x)
        return Model(inputs, out_embedding)
    else:
        outputs = Dense(class_nb, activation='softmax',name='softmax')(x)
        return Model(inputs, outputs)




