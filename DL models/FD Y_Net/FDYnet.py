# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 11:57:35 2021

@author: arumugaraj
"""

from tensorflow.keras.layers import Conv2D,Input,MaxPool2D, Activation,BatchNormalization,concatenate,Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from Dense import DenseBlock

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
    x = batchnorm_relu(x)
    x = Conv2D(filters=f, kernel_size=3, strides=1, padding='same')(x)
    x = batchnorm_relu(x)
    return x

def batchnorm_relu(inputs):
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    return x



def c1(x, filters):
    
    x = MaxPool2D(pool_size=2, strides=2, padding='valid')(x)
    x = DenseBlock(filters, x)
    return x

def c2(x, filters):
    
    x = DenseBlock(filters, x)
    x = Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding='valid')(x)

    return x

############ Model creation ###################


def FDY_net(I,f,R):
    
    
        #encoder2
    inputB = Input(shape=(256,2048,1))
    E1 = conv(inputB,f) #256, 2048, 32
    E2 = c1(E1,2*f)#128, 1024, 64
    E3 = c1(E2,4*f)#64, 512, 128
    E4 = c1(E3,8*f)#32, 256, 256
    E5 = c1(E4,8*f)#16, 128, 256

    
     # #Resize the sensor data
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

    
     #DECODER
    d1 = concatenate([R1,e5])
    C1 =Conv2D(f * 8, (1, 1), activation=None, padding="same")(d1)
    x = DenseBlock(f*8,C1)
    x = Conv2DTranspose(filters=8*f, kernel_size=2, strides=2, padding='valid')(C1)
    d2 = concatenate([x,R2,e4])
    C2 =Conv2D(f * 4, (1, 1), activation=None, padding="same")(d2)
    x = DenseBlock(f*4,C2)
    x = Conv2DTranspose(filters=4*f, kernel_size=2, strides=2, padding='valid')(x)
    d3 = concatenate([x,R3,e3])
    C3 =Conv2D(f * 2, (1, 1), activation=None, padding="same")(d3)
    x = DenseBlock(f*2,C3)
    x = Conv2DTranspose(filters=2*f, kernel_size=2, strides=2, padding='valid')(x)
    d4 = concatenate([x,R4,e2])
    C4 =Conv2D(f , (1, 1), activation=None, padding="same")(d4)
    x = DenseBlock(f,C4)
    x = Conv2DTranspose(filters=f, kernel_size=2, strides=2, padding='valid')(x)
    d5 = concatenate([x,R5,e1])
    d6 = conv(d5, 1)
   

    

    
    
    model = Model(inputs=[inputA,inputB], outputs=d6)
    model.compile(optimizer=Adam(learning_rate = 1e-4), loss=avg_NSR, metrics=[avg_log_SNR])
    return model



if __name__ == "__main__":
    I=(256, 256, 1);f=32;R=(256,2048,1)
    model = FDY_net(I,f,R)
    model.summary()
    