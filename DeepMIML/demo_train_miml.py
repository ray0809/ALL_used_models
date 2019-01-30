#coding=utf-8
import sys
import os
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.applications import ResNet50

from deepmiml import *
from data_loader import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def threshold(inp, thred=0.1):
    one = tf.ones_like(inp)
    zero = tf.zeros_like(inp)
    inp = tf.where(inp < thred, x=zero, y=one)
    return inp

def precision(y_true, y_pred):
    y_pred = threshold(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    y_pred = threshold(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall



if __name__ == "__main__":
    train_txt = sys.argv[1]
    test_txt = sys.argv[2]

    loss = "binary_crossentropy"
    nb_epoch = 10
    batch_size = 32
    l = 81 #
    k = 20
    model_name = "miml_resnet_16"
    base_model = ResNet50(include_top=False,pooling=None,input_shape=(224,224,3))
    # base_model.summary()
    # deepmiml = DeepMIML(L=L, K=K, base_model=base_model)
    deepmiml = create_miml_model(base_model, l, k)
    # deepmiml.load_weights('weights/best.h5')
    # deepmiml.model.summary()

    print("Compiling Deep MIML Model...")
    # adam = Adam(lr=0.0001)
    deepmiml.compile(optimizer='adadelta', loss=loss, metrics=[precision,recall])

    print("Start Training...")
    modelcheck = ModelCheckpoint('weights/20190128_best.h5',save_best_only=True,save_weights_only=False)
    train = dataLoader(train_txt,32,shuffle=True)
    test = dataLoader(test_txt,32,shuffle=False)
    deepmiml.fit_generator(train,
            steps_per_epoch=train.__len__(),
            epochs=nb_epoch,
            callbacks=[modelcheck],
            validation_data=test,
            validation_steps=test.__len__(),
            use_multiprocessing=True
            )

