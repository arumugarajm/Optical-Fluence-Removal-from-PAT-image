# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 18:12:31 2021

@author: arumugaraj
"""


from tensorflow.keras.layers import Conv2D,Add, BatchNormalization,UpSampling2D, Activation, Concatenate, Input
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


def BatchActivate(x, act=True):
    x = BatchNormalization()(x)
    if act == True:
        x = Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = BatchActivate(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = BatchActivate(shortcut, act=False)
    
    output = Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = BatchActivate(shortcut, act=False)
    
    output = Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = UpSampling2D((2, 2))(x)
    c = Concatenate()([u, xskip])
    return c


############ Model creation ###################

def ResUnet(image_size,channels):
    f = [16, 32, 64, 128, 256]
    inputs = Input((image_size, image_size, 1))
    
    ## Encoder
    E0 = inputs 
    E1 = stem(E0, f[0])
    E2 = residual_block(E1, f[1], strides=2)
    E3 = residual_block(E2, f[2], strides=2)
    E4 = residual_block(E3, f[3], strides=2)
    E5 = residual_block(E4, f[4], strides=2)
    
    ## Bridge
    B0 = conv_block(E5, f[4], strides=1)
    B1 = conv_block(B0, f[4], strides=1)
    
    ## Decoder
    U1 = upsample_concat_block(B1, E4) 
    D1 = residual_block(U1, f[4])
    U2 = upsample_concat_block(D1, E3)
    D2 = residual_block(U2, f[3])
    
    U3 = upsample_concat_block(D2, E2)
    D3 = residual_block(U3, f[2])
    
    U4 = upsample_concat_block(D3, E1)
    D4 = residual_block(U4, f[1])
    
    outputs = Conv2D(1, (1, 1), padding="same", activation="relu")(D4)
    # outputs = outputs + e0
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate = 1e-4), loss=avg_NSR, metrics=[avg_log_SNR])
    return model

if __name__ == "__main__":
    input_size = 256
    channels = 1
    model = ResUnet(input_size,channels)
    model.summary()

