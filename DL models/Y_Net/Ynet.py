#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 21:53:26 2021

@author: fistlab
"""

from tensorflow.keras.layers import Conv2D,Input,MaxPool2D, Activation,BatchNormalization,concatenate,Conv2DTranspose
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

def conv(x,f):
    x = Conv2D(filters=f, kernel_size=3, strides=1, padding='same')(x)
    x = Conv2D(filters=f, kernel_size=3, strides=1, padding='same')(x)
    return x

def batchnorm_relu(inputs):
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    return x


def c1(x, filters):
    
    x = MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = batchnorm_relu(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = batchnorm_relu(x)
    return x

def c2(x, filters):
    
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = batchnorm_relu(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = batchnorm_relu(x)
    x = Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding='valid')(x)
    return x
 
############ Model creation ###################

def Y_net(I,f,R):
    
    
        #encoder2
    inputB = Input(shape=(256,2048,1))
    E1 = conv(inputB,f) #256, 2048, 32
    E2 = c1(E1,2*f)#128, 1024, 64
    E3 = c1(E2,4*f)#64, 512, 128
    E4 = c1(E3,8*f)#32, 256, 256
    E5 = c1(E4,8*f)#16, 128, 256
    # E6 = c1(E5,16*f)#8, 64, 512 To increase the depth uncommand this line
    
    # #Resize the sensor data
    # R0 = Conv2D(filters=16*f, kernel_size=(1,8), strides=(1,8), padding='valid')(E6)
    R1 = Conv2D(filters=8*f, kernel_size=(1,8), strides=(1,8), padding='valid')(E5)#16, 16, 256
    R2 = Conv2D(filters=8*f, kernel_size=(1,8), strides=(1,8), padding='valid')(E4)
    R3 = Conv2D(filters=4*f, kernel_size=(1,8), strides=(1,8), padding='valid')(E3)
    R4 = Conv2D(filters=2*f, kernel_size=(1,8), strides=(1,8), padding='valid')(E2)
    R5 = Conv2D(filters=1*f, kernel_size=(1,8), strides=(1,8), padding='valid')(E1)
    
    
   
    #encoder1
    inputA = Input(shape=(256,256,1))
    e1 = conv(inputA,f) #256, 256, 32
    e2 = c1(e1,2*f) #128, 128, 64
    e3 = c1(e2,4*f) #64, 64, 128
    e4 = c1(e3,8*f) #32, 32, 256
    e5 = c1(e4,8*f)#16, 16, 256
    # e6 = c1(e5,16*f)#8, 8, 512 To increase the depth uncommand this line
    
    # #decoder
    # d0 = concatenate([R0,e6]) #To increase the depth uncommand this line
    # d01 = c2(d0, 16*f)
    d1 = concatenate([R1,e5])#16, 16, 512
    d2 = c2(d1, 8*f)#32, 32, 256 one
    d3 = concatenate([d2,R2,e4])
    d4 = c2(d3, 4*f) # two 64
    d5 = concatenate([d4,R3,e3])
    d6 = c2(d5, 2*f) #three 128
    d7 = concatenate([d6,R4,e2])
    d8 = c2(d7, f) #four 256
    d9 = concatenate([d8,R5,e1])
    d10 = conv(d9, 1)

    

    
    
    model = Model(inputs=[inputA,inputB], outputs=d10)
    model.compile(optimizer=Adam(learning_rate = 1e-4), loss=avg_NSR, metrics=[avg_log_SNR])
    return model



if __name__ == "__main__":
    I=(256, 256, 1);f=32;R=(256,2048,1)
    model = Y_net(I,f,R)
    model.summary()
    