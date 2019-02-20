#coding=utf-8
import keras.backend as K
from keras.models import Model
from keras.layers import *
from keras.applications import *
from keras.regularizers import *
from resnet import *


def Normalize(x):
    x = K.l2_normalize(x, axis=-1)
    return x

def Distance(x, margin=1):
    q, p, n = x[0], x[1], x[2]
    dis_qp = K.sum(K.square(q - p), axis=-1, keepdims=True)
    dis_qn = K.sum(K.square(q - n), axis=-1, keepdims=True)
    loss = K.maximum(dis_qp - dis_qn + margin, 0.0)
    return loss



def vgg(dis):
    """
    Triplet Loss的基础网络，可以替换其他网络结构
    """
    inputs = Input(shape=(None,None,3))
    base_model = VGG16(include_top=False, pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False
    x = Lambda(vgg16.preprocess_input, arguments={'mode':'tf'})(inputs)
    x = base_model(x)
    x = Dropout(0.5)(x)
    x = Dense(128,activation='relu')(x)
    if dis == 'cos':
        x = Lambda(Normalize)(x)
    model = Model(inputs, x)
    return model

def TripletNet(margin=1, dis='euclidean'):
    base_model = vgg(dis)
    base_model.summary()
    input_q = Input(shape=(None, None, 3))
    input_p = Input(shape=(None, None, 3))
    input_n = Input(shape=(None, None, 3))

    encode_q = base_model(input_q)
    encode_p = base_model(input_p)
    encode_n = base_model(input_n)

    
    if dis == 'cos':
        # cos distance
        dis_q_p = Dot(axes=-1, normalize=False)([encode_q,encode_p])
        dis_q_n = Dot(axes=-1, normalize=False)([encode_q,encode_n])
        loss = Lambda(lambda x: K.relu(margin+x[1]-x[0]))([dis_q_p,dis_q_n])
        
    else:
        # euclidean distance
        loss = Lambda(Distance,arguments={'margin':margin})([encode_q,encode_p,encode_n])

    model_train = Model([input_q,input_p,input_n],loss)
    model_test = Model(input_q,encode_q)
    return model_train, model_test