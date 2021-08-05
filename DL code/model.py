# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 17:22:25 2021

@author: arumugaraj
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, Input, concatenate, ReLU, MaxPool2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import math
import numpy as np


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
    #Y = PReLU()(Y)
    return Y



def U_Net(input_size, input_channels, filters):

    # set up input
    X = Input(shape=(input_size, input_size, input_channels))

    # convolution pre-processing
    Y = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(X)
    Y = BatchActivate(Y)

    # convolutions 1
    Y = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(Y)
    Y = BatchActivate(Y)

    Y = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(Y)
    Y_skip_1 = BatchActivate(Y)
   
    

    # pooling 1
    Y = MaxPool2D(pool_size=2, strides=2, padding='valid')(Y_skip_1)

    # convolutions 2
    Y = Conv2D(filters=2*filters, kernel_size=3, strides=1, padding='same')(Y)
    Y = BatchActivate(Y)

    Y = Conv2D(filters=2*filters, kernel_size=3, strides=1, padding='same')(Y)
    Y_skip_2 = BatchActivate(Y)
    

    # pooling 2
    Y = MaxPool2D(pool_size=2, strides=2, padding='valid')(Y_skip_2)

    # convolutions 3
    Y = Conv2D(filters=4*filters, kernel_size=3, strides=1, padding='same')(Y)
    Y = BatchActivate(Y)
    
    
    Y = Conv2D(filters=4*filters, kernel_size=3, strides=1, padding='same')(Y)
    Y_skip_3 = BatchActivate(Y)
   

    # pooling 3
    Y = MaxPool2D(pool_size=2, strides=2, padding='valid')(Y_skip_3)

    # convolutions 4
    Y = Conv2D(filters=8*filters, kernel_size=3, strides=1, padding='same')(Y)
    Y = BatchActivate(Y)

    Y = Conv2D(filters=8*filters, kernel_size=3, strides=1, padding='same')(Y)
    Y_skip_4 = BatchActivate(Y)
    

    # pooling 4
    Y = MaxPool2D(pool_size=2, strides=2, padding='valid')(Y_skip_4)

    # convolutions 5
    Y = Conv2D(filters=16*filters, kernel_size=3, strides=1, padding='same')(Y)
    Y = BatchActivate(Y)

    Y = Conv2D(filters=16*filters, kernel_size=3, strides=1, padding='same')(Y)
    Y = BatchActivate(Y)

    # upconvolution and concatenation 1
    Y = Conv2DTranspose(filters=8*filters, kernel_size=2, strides=2, padding='valid')(Y)
    Y = BatchActivate(Y)

    Y = concatenate([Y_skip_4, Y])

    # convolutions 6
    Y = Conv2D(filters=8*filters, kernel_size=3, strides=1, padding='same')(Y)
    Y = BatchActivate(Y)

    Y = Conv2D(filters=8*filters, kernel_size=3, strides=1, padding='same')(Y)
    Y = BatchActivate(Y)

    # upconvolution and concatenation 2
    Y = Conv2DTranspose(filters=4*filters, kernel_size=2, strides=2, padding='valid')(Y)
    Y = BatchActivate(Y)
    

    Y = concatenate([Y_skip_3, Y])
    

    # convolutions 7
    Y = Conv2D(filters=4*filters, kernel_size=3, strides=1, padding='same')(Y)
    Y = BatchActivate(Y)

    Y = Conv2D(filters=4*filters, kernel_size=3, strides=1, padding='same')(Y)
    Y = BatchActivate(Y)

    # upconvolution and concatenation 3
    Y = Conv2DTranspose(filters=2*filters, kernel_size=2, strides=2, padding='valid')(Y)
    Y = BatchActivate(Y)

    Y = concatenate([Y_skip_2, Y])

    # convolutions 8
    Y = Conv2D(filters=2*filters, kernel_size=3, strides=1, padding='same')(Y)
    Y = BatchActivate(Y)

    Y = Conv2D(filters=2*filters, kernel_size=3, strides=1, padding='same')(Y)
    Y = BatchActivate(Y)

    # upconvolution and concatenation 4
    Y = Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding='valid')(Y)
    Y = BatchActivate(Y)

    Y = concatenate([Y_skip_1, Y])

    # convolutions 9
    Y = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(Y)
    Y = BatchActivate(Y)

    Y = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(Y)
    Y = BatchActivate(Y)

    # convolution post-processing
    Y = Conv2D(filters=input_channels, kernel_size=1, strides=1, padding='same')(Y)

    # final concatenation and convolution
    Y = X+Y

    # set up model and compile
    model = Model(inputs=X, outputs=Y)
    model.compile(optimizer=Adam(lr=0.001), loss=avg_NSR, metrics=[avg_log_SNR])
    # model.summary()
    

    # return model
    return model



    
    
