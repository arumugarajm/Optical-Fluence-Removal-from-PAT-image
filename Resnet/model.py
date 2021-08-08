# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 15:39:33 2021

@author: arumugaraj
"""

import tensorflow as tf

from tensorflow.keras.layers import Conv2D,Activation, BatchNormalization, Conv2DTranspose, Input, concatenate, ELU, MaxPool2D,LeakyReLU,ReLU,PReLU
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import math
import numpy as np
from keras.layers import add




# # function for creating an identity or projection residual module
# def RB(layer_in, n_filters):
# 	merge_input = layer_in
# 	# check if the number of filters needs to be increase, assumes channels last format
# 	if layer_in.shape[-1] != n_filters:
# 		merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
# 	# conv1
# 	conv1 = Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
# 	# conv2
# 	conv2 = Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
# 	# add filters, assumes filters/channels last
# 	layer_out = add([conv2, merge_input])
# 	# activation function
# 	layer_out = Activation('relu')(layer_out)
# 	return layer_out


def avg_NSR(Y_true, Y_pred):
    signal = tf.norm(tf.reshape(Y_true, [tf.shape(Y_true)[0], -1]), axis=1)
    noise = tf.norm(tf.reshape(Y_true - Y_pred, [tf.shape(Y_true)[0], -1]), axis=1)

    return tf.math.reduce_mean(noise / signal)

def avg_log_SNR(Y_true, Y_pred):
    signal = tf.norm(tf.reshape(Y_true, [tf.shape(Y_true)[0], -1]), axis=1)
    noise = tf.norm(tf.reshape(Y_true - Y_pred, [tf.shape(Y_true)[0], -1]), axis=1)

    return tf.math.reduce_mean(20. * tf.math.log(signal / noise) / tf.math.log(10.0))


def BatchActivate(Y):
    Y = BatchNormalization()(Y)
    Y = Activation('relu')(Y)
    # Y = PReLU()(Y)
    return Y
def RB(inputs,filters):
    # convolutions RB1
    inputs=inputs
    filters=filters
    Y = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(inputs)
    Y = BatchActivate(Y)
    Y = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(Y)
    # Y = BatchActivate(Y)
    # Y =  BatchNormalization()(Y)
    Y=add([inputs,Y])
    results=BatchActivate(Y)
    return results
    

def RNet(input_size, input_channels, filters):

    # set up input
    X = Input(shape=(input_size, input_size, input_channels))

    # convolution
    Y = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(X)
    Y = BatchActivate(Y)

    # convolutions RB1
    Y1 = RB(Y,32)
 
    
    # convolutions RB2
    Y2 = RB(Y1,32)
 
        # convolutions RB3
    Y3 = RB(Y2,32)
   
        # convolutions RB4
    Y4 = RB(Y3,32)
  
    #     # convolutions RB5
    # Y = RB(Y,32)
    #     # convolutions RB6
    # Y = RB(Y,32)
    #     # convolutions RB7
    # Y = RB(Y,32)
    #     # convolutions RB8
    # Y = RB(Y,32)
    #     # convolutions RB9
    # Y = RB(Y,32)
    #     # convolutions RB10
    # Y = RB(Y,32)
    #     # convolutions RB11
    # Y = RB(Y,32)
    #     # convolutions RB12
    # Y = RB(Y,32)
    #     # convolutions RB13
    # Y = RB(Y,32)
    #     # convolutions RB14
    # Y = RB(Y,32)
    #     # convolutions RB15
    # Y = RB(Y,32)   
    # # convolutions RB16
    # Y = RB(Y,32)
    #     # convolutions RB17
    # Y = RB(Y,32)
    #     # convolutions RB18
    # Y = RB(Y,32)
    #     # convolutions RB19
    # Y = RB(Y,32)
    #     # convolutions RB20
    # Y = RB(Y,32)   
    # # convolutions RB21
    # Y = RB(Y,32)
    #     # convolutions RB22
    # Y = RB(Y,32)
    #     # convolutions RB23
    # Y = RB(Y,32)
    #     # convolutions RB24
    # Y = RB(Y,32)
    #     # convolutions RB25
    # Y = RB(Y,32)
    #     # convolutions RB26
    # Y = RB(Y,32)
    #     # convolutions RB27
    # Y = RB(Y,32)
    #     # convolutions RB28
    # Y = RB(Y,32)
    #     # convolutions RB29
    # Y = RB(Y,32)
    #     # convolutions RB30
    # Y = RB(Y,32)
    #     # convolutions RB31
    # Y = RB(Y,32)
    #     # convolutions RB32
    # Y = RB(Y,32)
    
        # convolution
    Y = Conv2D(filters=2*filters, kernel_size=3, strides=1, padding='same')(Y4)
    Y = BatchActivate(Y)
    
        # convolution
    Y = Conv2D(filters=2*filters, kernel_size=3, strides=1, padding='same')(Y)
    Y = BatchActivate(Y)


    # convolution
    Y = Conv2D(filters=1, kernel_size=1, strides=1, padding='same')(Y)
    Y = BatchActivate(Y)
   # set up model and compile
    model = Model(inputs=X, outputs=Y)
    model.compile(optimizer=Adam(lr=0.0001), loss='mse', metrics=["mse"])
    # model.summary()
    

    # return model
    return model



    
    