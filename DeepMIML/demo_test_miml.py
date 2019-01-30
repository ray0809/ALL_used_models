#codinutf-8
import sys
import os
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import precision_score, recall_score
from keras.applications import ResNet50
from keras.preprocessing import image

from deepmiml import *

if __name__ == '__main__':
    test = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    y_true = [int(i) for i in test.split()]
    path = sys.argv[1]
    img = image.load_img(path, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, 0)

    base_model = ResNet50(include_top=False,pooling=None,input_shape=(224,224,3))
    # base_model.summary()
    miml_model = create_miml_model(base_model, 81, 20)
    miml_model.load_weights('weights/best.h5')
    # deep_miml = DeepMIML(81, 20, model=miml_model)
    pred = miml_model.predict(img)[0]
    pred[pred>0.1] = 1
    pred[pred<0.1] = 0
    pred = list(pred.astype('int'))
    # print(np.where(pred > 0.1))
    # print('pred',pred)
    # print('y_true',y_true)
    print('recall:',recall_score(y_true,pred))
    print('precision:',precision_score(y_true,pred))

