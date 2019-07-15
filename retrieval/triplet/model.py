#coding=utf-8
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import *
# from keras.applications import *
from keras.regularizers import *
import keras_applications

from classification_models import Classifiers


def triplet_loss(x, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    anchor, positive, negative = x[0], x[1], x[2]
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
      
    return loss

def Edu_distance(x, margin=1):
    q, p, n = x[0], x[1], x[2]
    dis_qp = K.sum(K.square(q - p + 1e-8), axis=-1, keepdims=True)
    dis_qn = K.sum(K.square(q - n + 1e-8), axis=-1, keepdims=True)
    loss = K.relu(dis_qp - dis_qn + margin + 1e-8)
    return K.mean(loss)

def Cos_distance(x, margin=0.3):
    q, p, n = x[0], x[1], x[2]
    dis_q_p = Dot(axes=-1, normalize=True)([q, p])
    dis_q_n = Dot(axes=-1, normalize=True)([q,  n])
    loss = K.relu(margin + dis_q_p - dis_q_n + 1e-8)
    return K.mean(loss)

def resnet34():
    classifier, preprocess_input = Classifiers.get('resnet34')
    base_model = classifier((None, None, 3), weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128)(x)    
    model = Model(base_model.input, x)
    return model


# def resnet50():
#     keras_applications.resnet50

def vgg():
    """
    Triplet Loss的基础网络，可以替换其他网络结构
    """
    inputs = Input(shape=(None,None,3))
    base_model = VGG16(include_top=False, pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False
    x = Lambda(vgg16.preprocess_input, arguments={'mode':'tf'})(inputs)
    x = base_model(x)
    # x = Dropout(0.5)(x)
    x = Dense(128)(x)
    # out = Lambda(Normalize)(x)
    model = Model(inputs, x)
    return model

def TripletNet(margin=1, dis='euclidean'):
    # base_model = vgg()
    base_model = resnet34()
    # base_model.summary()
    input_q = Input(shape=(None, None, 3))
    input_p = Input(shape=(None, None, 3))
    input_n = Input(shape=(None, None, 3))

    encode_q = base_model(input_q)
    encode_p = base_model(input_p)
    encode_n = base_model(input_n)

    
    loss = triplet_loss([encode_q, encode_p, encode_n], alpha=margin)

    model_train = Model([input_q,input_p,input_n],[encode_q, encode_p, encode_n])
    model_train.add_loss(loss)
    model_test = Model(input_q,encode_q)
    return model_train, model_test