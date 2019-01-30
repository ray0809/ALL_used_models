#coding=utf-8
from keras.models import Model
from keras.layers import Input,Flatten,Dropout,Dense,Lambda
from keras.applications import resnet50



def crtModel(class_nb, freezen=True):
    base_model = resnet50.ResNet50(include_top=False,pooling='avg')
    if freezen:
        for layer in base_model.layers:
            layer.trainable = False
    inputs = Input(shape=(None,None,3))
    x = Lambda(resnet50.preprocess_input,arguments={'mode':'tf'})(inputs)
    x = base_model(x)
    # x = Flatten()(x)
    # x = Dropout(0.5)(x)
    embedding = Dense(512,name='embedding')(x)
    outputs = Dense(class_nb, activation='softmax',name='softmax')(embedding)
    model = Model(inputs, outputs)
    return model




