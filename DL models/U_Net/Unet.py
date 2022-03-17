# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:51:34 2021

@author: arumugaraj
"""



from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, Input, concatenate, ReLU, MaxPool2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf



########## Loss fuction to train the model ##################
def avg_NSR(Y_true, Y_pred):
    signal = tf.norm(tf.reshape(Y_true, [tf.shape(Y_true)[0], -1]), axis=1)
    noise = tf.norm(tf.reshape(Y_true - Y_pred, [tf.shape(Y_true)[0], -1]), axis=1)

    return tf.math.reduce_mean(noise / signal)


########## Metric function to monitor ########################
def avg_log_SNR(Y_true, Y_pred):
    signal = tf.norm(tf.reshape(Y_true, [tf.shape(Y_true)[0], -1]), axis=1)
    noise = tf.norm(tf.reshape(Y_true - Y_pred, [tf.shape(Y_true)[0], -1]), axis=1)

    return tf.math.reduce_mean(20. * tf.math.log(signal / noise) / tf.math.log(10.0))

####### predefined layers for the model creation #######################

def BatchActivate(Y):
    Y = BatchNormalization()(Y)
    Y = Activation('relu')(Y)
    #Y = PReLU()(Y)
    return Y

############ Model creation ###################

def Unet(input_size, input_channels, filters):

    # set up input
    X = Input(shape=(input_size, input_size, input_channels))
    
    ### Encoder path

    # convolution pre-processing
    E = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(X)
    E = BatchActivate(E)

    # convolutions 1
    E = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(E)
    E = BatchActivate(E)

    E = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(E)
    E_skip_1 = BatchActivate(E)
   
    

    # pooling 1
    E = MaxPool2D(pool_size=2, strides=2, padding='valid')(E_skip_1)

    # convolutions 2
    E = Conv2D(filters=2*filters, kernel_size=3, strides=1, padding='same')(E)
    E = BatchActivate(E)

    E = Conv2D(filters=2*filters, kernel_size=3, strides=1, padding='same')(E)
    E_skip_2 = BatchActivate(E)
    

    # pooling 2
    E = MaxPool2D(pool_size=2, strides=2, padding='valid')(E_skip_2)

    # convolutions 3
    E = Conv2D(filters=4*filters, kernel_size=3, strides=1, padding='same')(E)
    E = BatchActivate(E)
    
    
    E = Conv2D(filters=4*filters, kernel_size=3, strides=1, padding='same')(E)
    E_skip_3 = BatchActivate(E)
   

    # pooling 3
    E = MaxPool2D(pool_size=2, strides=2, padding='valid')(E_skip_3)

    # convolutions 4
    E = Conv2D(filters=8*filters, kernel_size=3, strides=1, padding='same')(E)
    E = BatchActivate(E)

    E = Conv2D(filters=8*filters, kernel_size=3, strides=1, padding='same')(E)
    E_skip_4 = BatchActivate(E)
    

    # # pooling 4
    E = MaxPool2D(pool_size=2, strides=2, padding='valid')(E_skip_4)

    # convolutions 5
    E = Conv2D(filters=16*filters, kernel_size=3, strides=1, padding='same')(E)
    E = BatchActivate(E)

    E = Conv2D(filters=16*filters, kernel_size=3, strides=1, padding='same')(E)
    E = BatchActivate(E)
    
    
    
    ### Decoder path

    # # upconvolution and concatenation 1
    D = Conv2DTranspose(filters=8*filters, kernel_size=2, strides=2, padding='valid')(E)
    D = BatchActivate(D)

    D = concatenate([E_skip_4, D])

    # convolutions 6
    D = Conv2D(filters=8*filters, kernel_size=3, strides=1, padding='same')(D)
    D = BatchActivate(D)

    D = Conv2D(filters=8*filters, kernel_size=3, strides=1, padding='same')(D)
    D = BatchActivate(D)

    # upconvolution and concatenation 2
    D = Conv2DTranspose(filters=4*filters, kernel_size=2, strides=2, padding='valid')(D)   #(D)
    D = BatchActivate(D)
    

    D = concatenate([E_skip_3, D])
    

    # convolutions 7
    D = Conv2D(filters=4*filters, kernel_size=3, strides=1, padding='same')(D)
    D = BatchActivate(D)

    D = Conv2D(filters=4*filters, kernel_size=3, strides=1, padding='same')(D)
    D = BatchActivate(D)

    # upconvolution and concatenation 3
    D = Conv2DTranspose(filters=2*filters, kernel_size=2, strides=2, padding='valid')(D)
    D = BatchActivate(D)

    D = concatenate([E_skip_2, D])

    # convolutions 8
    D = Conv2D(filters=2*filters, kernel_size=3, strides=1, padding='same')(D)
    D = BatchActivate(D)

    D = Conv2D(filters=2*filters, kernel_size=3, strides=1, padding='same')(D)
    D = BatchActivate(D)

    # upconvolution and concatenation 4
    D = Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding='valid')(D)
    D = BatchActivate(D)

    D = concatenate([E_skip_1, D])

    # convolutions 9
    D = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(D)
    D = BatchActivate(D)

    D = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(D)
    D = BatchActivate(D)

    # convolution post-processing
    D = Conv2D(filters=input_channels, kernel_size=1, strides=1, padding='same')(D)

    # final concatenation and convolution
    # D = X+D

    # set up model and compile
    model = Model(inputs=X, outputs=D)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=avg_NSR, metrics=[avg_log_SNR])
    model.summary()
    

    return model



if __name__ == "__main__":
    model = Unet(input_size=256, input_channels=1, filters=64)
    model.summary()




