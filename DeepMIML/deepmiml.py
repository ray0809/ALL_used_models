import keras.backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Reshape, Permute, Activation,Flatten,Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D


MIML_FIRST_LAYER_NAME = "miml/first_layer"
MIML_CUBE_LAYER_NAME = "miml/cube"
MIML_TABLE_LAYER_NAME = "miml/table"
MIML_OUTPUT_LAYER_NAME = "miml/output"


def create_miml_model(base_model, l, k, name="miml", fine_tune=False):
    """
    Arguments:
        base_model (Sequential):
            A Neural Network in keras form (e.g. VGG, GoogLeNet)
        L (int):
            number of labels

    """
    # model = Sequential(layers=base_model.layers, name=name)
    if fine_tune:
        for layer in base_model.layers:
            layer.trainable = False
    base_in = base_model.input
    base_out = base_model.output
    _, H, W, C = K.int_shape(base_out)
    n_instances = H * W
    x = Dropout(0.5)(base_out)
    x = Reshape((n_instances, 1, C))(x)
    # x = Convolution2D(l*k,(1,1))(x)
    # x = Reshape((n_instances, k, l))(x)
    # x = MaxPooling2D((1, k),strides=(1,1))(x)
    x = Convolution2D(l,(1,1))(x)
    x = Activation("softmax")(x)
    x = MaxPooling2D((n_instances,1),strides=(1,1))(x)
    x = Flatten()(x)

    model = Model(base_in, x)
    return model


