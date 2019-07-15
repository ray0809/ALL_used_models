#codng=utf-8
import keras
import keras.backend as K
from keras.layers import *
from keras.models import *
from keras.regularizers import *




def resnet_block(inputs,num_filters=16,
                  kernel_size=3,strides=1,
                  activation='relu'):
    x = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',
           kernel_initializer='TruncatedNormal',kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    if(activation):
        x = Activation('relu')(x)
    return x


# 建一个20层的ResNet网络 
def resnet_v1(input_shape):
    #input_shape should be (width,height,channel)
    inputs = Input(shape=input_shape)# Input层，用来当做占位使用

    #第一层
    x = resnet_block(inputs)
    # print('layer1,xshape:',x.shape)
    # 第2~7层
    for i in range(6):
        a = resnet_block(inputs = x)
        b = resnet_block(inputs=a,activation=None)
        x = keras.layers.add([x,b])
        x = Activation('relu')(x)
    # out：32*32*16
    # 第8~13层
    for i in range(6):
        if i == 0:
            a = resnet_block(inputs = x,strides=2,num_filters=32)
        else:
            a = resnet_block(inputs = x,num_filters=32)
        b = resnet_block(inputs=a,activation=None,num_filters=32)
        if i==0:
            x = Conv2D(32,kernel_size=3,strides=2,padding='same',
                       kernel_initializer='TruncatedNormal',kernel_regularizer=l2(1e-4))(x)
        x = keras.layers.add([x,b])
        x = Activation('relu')(x)

    for i in range(6):
        if i ==0 :
            a = resnet_block(inputs = x,strides=2,num_filters=64)
        else:
            a = resnet_block(inputs = x,num_filters=64)

        b = resnet_block(inputs=a,activation=None,num_filters=64)
        if i == 0:
            x = Conv2D(64,kernel_size=3,strides=2,padding='same',
                       kernel_initializer='TruncatedNormal',kernel_regularizer=l2(1e-4))(x)
        x = keras.layers.add([x,b])# 相加操作，要求x、b shape完全一致
        x = Activation('relu')(x)
  
    y = GlobalAveragePooling2D()(x)
    model = Model(inputs=inputs,outputs=y)
    return model

if __name__ == '__main__':
    model=resnet_v1((224,224,3))
    model.summary()