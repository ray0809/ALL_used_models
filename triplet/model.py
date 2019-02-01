#coding=utf-8
import keras.backend as K
from keras.models import Model
from keras.layers import Input,Flatten,Dropout,Dense,Lambda,Dot
from keras.applications import resnet50

def Normalize(x):
    x = K.l2_normalize(x, axis=-1)
    return x

def Distance(x, margin=1):
    q, p, n = x[0], x[1], x[2]
    dis_qp = K.square(q - p)
    dis_qn = K.square(q - n)
    loss = K.relu(dis_qp - dis_qn + margin)
    return loss


def crtModel(freezen=False):
    base_model = resnet50.ResNet50(include_top=False,pooling='avg')
    if freezen:
        for layer in base_model.layers:
            layer.trainable = False
    inputs = Input(shape=(None,None,3))
    x = Lambda(resnet50.preprocess_input,arguments={'mode':'tf'})(inputs)
    x = base_model(x)
    x = Dense(512,name='embedding')(x)
    embedding = Lambda(Normalize, name='l2_embedding')(x)
    model = Model(inputs, embedding)
    # model.summary()
    return model


def TripletNet(margin=1, freezen=False):
    base_model = crtModel(freezen=freezen)
    base_model.summary()
    # base_model.load_weights('weights/20190130_full_8583.h5',by_name=True)
    input_q = Input(shape=(None, None, 3))
    input_p = Input(shape=(None, None, 3))
    input_n = Input(shape=(None, None, 3))

    encode_q = base_model(input_q)
    encode_p = base_model(input_p)
    encode_n = base_model(input_n)

    dis_q_p = Dot(axes=-1, normalize=False)([encode_q,encode_p])
    dis_q_n = Dot(axes=-1, normalize=False)([encode_q,encode_n])

    # cos distance
    loss = Lambda(lambda x: K.relu(margin+x[1]-x[0]))([dis_q_p,dis_q_n])
    # absolute distance
    # loss = Lambda(Distance,arguments={'margin':margin})([encode_q,encode_p,encode_n])

    model_train = Model([input_q,input_p,input_n],loss)
    model_test = Model(input_q,encode_q)
    return model_train, model_test