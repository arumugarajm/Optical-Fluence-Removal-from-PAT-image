# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 14:21:01 2021

@author: arumugaraj
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Flatten,Dense,UpSampling2D, Concatenate,Add, BatchNormalization,Activation,LeakyReLU, Conv2DTranspose, Input, concatenate,  MaxPool2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
import keras.backend as K
import glob
import scipy.io

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


def Act(x):
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # x = Activation("relu")(x)
    return x
def conv(x,filters):
    
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = Act(x)
    p = MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    return x,p

def conv1(x,skip,filters):
    x =  Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same')(x)
    x = concatenate([x,skip])
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    
    return x


def bn_act(x, act=True):
    x = BatchNormalization()(x)
    if act == True:
        x = Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = UpSampling2D((2, 2))(x)
    c = Concatenate()([u, xskip])
    return c

def conv2(x,filters):
    x = Conv2D(filters=filters, kernel_size=4, strides=2, padding='same')(x)
    x = Act(x)
    return x
    

############ Model creation ###################

############ Generator Model creation ###################
 
def Generator(image_size):
    f = [16, 32, 64, 128, 256]
    inputs = Input((image_size))
    
     ## Encoder
    e0 = inputs 
    # print(e0.shape)
    e1 = stem(e0, f[0])
    # print(e1.shape)
    e2 = residual_block(e1, f[1], strides=2)
    # print(e2.shape)
    e3 = residual_block(e2, f[2], strides=2)
    # print(e3.shape)
    e4 = residual_block(e3, f[3], strides=2)
    # print(e4.shape)
    e5 = residual_block(e4, f[4], strides=2)
    # print(e5.shape)
    
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    # print(b0.shape)
    b1 = conv_block(b0, f[4], strides=1)
    # print(b1.shape)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e4) 
    d1 = residual_block(u1, f[4])
    # print(u1.shape,d1.shape)
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    # print(u2.shape,d2.shape)
    
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    # print(u3.shape,d3.shape)
    
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    # print(u4.shape,d4.shape)
    
    outputs = Conv2D(1, (1, 1), padding="same", activation="relu")(d4)#208, 208, 1
    # outputs = outputs + e0
    # print(outputs.shape)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate = 1e-4), loss=avg_NSR, metrics=[avg_log_SNR])
    return model




############ Discriminator Model creation ###################

def Discriminator(image_shape):
    filters = 64
    # input_sized = (256,256,1)
    
    inputs = Input(shape=(image_shape))
    
    x = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(inputs)
    x = LeakyReLU(alpha = 0.2)(x)
    # print(inputs.shape)
    x = conv2(inputs,filters)
    # print(x.shape)
    x = conv2(x,2*filters)
    x = conv2(x,4*filters)
    x = conv2(x,8*filters)
    
    
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha = 0.2)(x)
       
    x = Dense(1)(x)
    validity = Activation('sigmoid')(x)
    
    model = Model(inputs = inputs, outputs = validity)
    opt = Adam(learning_rate=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

    
if __name__ == "__main__":
    model = Generator((256,256,1))
    model1 = Discriminator((256,256,1))
    model.summary()
    model1.summary()

############### perceptual loss function ##############

# def vgg_loss(y_true, y_pred):
    
#     image_shape = (384,384,3)
#     vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
#     vgg19.trainable = False
#         # Make trainable as False
#     for l in vgg19.layers:
#         l.trainable = False
#     model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
#     model.trainable = False
    
#     return K.mean(K.square(model(y_true) - model(y_pred)))

    
    

    
       