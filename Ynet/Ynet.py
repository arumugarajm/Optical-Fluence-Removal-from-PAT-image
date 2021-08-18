# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 23:50:12 2021

@author: arumugaraj
"""

from tensorflow.keras.layers import Conv2D,Input,MaxPool2D, Activation,BatchNormalization,concatenate,Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
def avg_NSR(Y_true, Y_pred):
    signal = tf.norm(tf.reshape(Y_true, [tf.shape(Y_true)[0], -1]), axis=1)
    noise = tf.norm(tf.reshape(Y_true - Y_pred, [tf.shape(Y_true)[0], -1]), axis=1)

    return tf.math.reduce_mean(noise / signal)

def avg_log_SNR(Y_true, Y_pred):
    signal = tf.norm(tf.reshape(Y_true, [tf.shape(Y_true)[0], -1]), axis=1)
    noise = tf.norm(tf.reshape(Y_true - Y_pred, [tf.shape(Y_true)[0], -1]), axis=1)

    return tf.math.reduce_mean(20. * tf.math.log(signal / noise) / tf.math.log(10.0))



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
 

def Y_net(I,f,R):
    
    
        #encoder2
    inputB = Input(shape=(256,512,1))
    E1 = conv(inputB,f)
    # print(E1.shape)
    E2 = c1(E1,2*f)
    E3 = c1(E2,4*f)
    E4 = c1(E3,8*f)
    E5 = c1(E4,8*f)#16, 32, 256
    # print(E5.shape)
    
    #Resize
    R1 = Conv2D(filters=8*f, kernel_size=(1,2), strides=(1,2), padding='valid')(E5)#16, 16, 256
    # print(R1.shape)
    R2 = Conv2D(filters=8*f, kernel_size=(1,2), strides=(1,2), padding='valid')(E4)
    # print(R2.shape)
    R3 = Conv2D(filters=4*f, kernel_size=(1,2), strides=(1,2), padding='valid')(E3)
    R4 = Conv2D(filters=2*f, kernel_size=(1,2), strides=(1,2), padding='valid')(E2)
    R5 = Conv2D(filters=1*f, kernel_size=(1,2), strides=(1,2), padding='valid')(E1)
    
    
   
    #encoder1
    inputA = Input(shape=(256,256,1))
    e1 = conv(inputA,f)
    e2 = c1(e1,2*f)
    e3 = c1(e2,4*f)
    e4 = c1(e3,8*f)
    # print(e4.shape)
    e5 = c1(e4,8*f)#16,16,256
    # print(e5.shape)
    
    #decoder
    d1 = concatenate([R1,e5])#16, 16, 512
    # print(d1.shape)
    d2 = c2(d1, 8*f)#32, 32, 256 one
    # print(d2.shape)
    d3 = concatenate([d2,R2,e4])
    # print(d3.shape)
    d4 = c2(d3, 4*f) # two 64
    # print(d4.shape)
    d5 = concatenate([d4,R3,e3])
    # print(d5.shape)
    d6 = c2(d5, 2*f) #three 128
    d7 = concatenate([d6,R4,e2])
    # print(d7.shape)
    d8 = c2(d7, f) #four 256
    # print(d8.shape)
    d9 = concatenate([d8,R5,e1])
    # print(d9.shape)
    d10 = conv(d9, 1)
    # print(d10.shape)

    

    
    
    model = Model(inputs=[inputA,inputB], outputs=d10)
    model.compile(optimizer=Adam(lr = 1e-4), loss = 'mse', metrics = ['mse'])
    return model



if __name__ == "__main__":
    model = Y_net((256, 256, 1),32,(256,512,1))
    model.summary()
    
    








