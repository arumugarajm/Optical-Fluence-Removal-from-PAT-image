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
    # x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    # x = batchnorm_relu(x)
    # x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    # x = batchnorm_relu(x)
    x = DenseBlock(filters * 2, x)
    return x

def c2(x, filters):
    
    # x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    # x = batchnorm_relu(x)
    # x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    # x = batchnorm_relu(x)
    x = DenseBlock(filters * 2, x)
    x = Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding='valid')(x)
    # x = DenseBlock(filters * 2, x)
    return x


def Y_net(I,f,R):
    
    
        #encoder2
    inputB = Input(shape=(256,2048,1))
    E1 = conv(inputB,f) #256, 2048, 32
    # print(E1.shape)
    E2 = c1(E1,2*f)#128, 1024, 64
    # print(E2.shape)
    E3 = c1(E2,4*f)#64, 512, 128
    # print(E3.shape)
    E4 = c1(E3,8*f)#32, 256, 256
    # print(E4.shape)
    E5 = c1(E4,8*f)#16, 128, 256
    # print(E5.shape)
    E6 = c1(E5,16*f)#8, 64, 512
    print(E6.shape)
    
    
     # #Resize
    R0 = Conv2D(filters=16*f, kernel_size=(1,8), strides=(1,8), padding='valid')(E6)
    R1 = Conv2D(filters=8*f, kernel_size=(1,8), strides=(1,8), padding='valid')(E5)#16, 16, 256
    # print(R1.shape)
    R2 = Conv2D(filters=8*f, kernel_size=(1,8), strides=(1,8), padding='valid')(E4)
    # print(R2.shape)
    R3 = Conv2D(filters=4*f, kernel_size=(1,8), strides=(1,8), padding='valid')(E3)
    R4 = Conv2D(filters=2*f, kernel_size=(1,8), strides=(1,8), padding='valid')(E2)
    R5 = Conv2D(filters=1*f, kernel_size=(1,8), strides=(1,8), padding='valid')(E1)
    
    
   
    #encoder1
    inputA = Input(shape=(256,256,1))
    e1 = conv(inputA,f) #256, 256, 32
    # print(e1.shape)
    e2 = c1(e1,2*f) #128, 128, 64
    # print(e2.shape)
    e3 = c1(e2,4*f) #64, 64, 128
    # print(e3.shape)
    e4 = c1(e3,8*f) #32, 32, 256
    # print(e4.shape)
    e5 = c1(e4,8*f)#16, 16, 256
    # print(e5.shape)
    e6 = c1(e5,16*f)#8, 8, 512
    print(e6.shape)
    
    # #decoder
    d0 = concatenate([R0,e6])
    print(d0.shape)
    d01 = c2(d0, 16*f)
    d1 = concatenate([d01,R1,e5])#16, 16, 512
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
    I=(256, 256, 1);f=32;R=(256,2048,1)
    model = Y_net(I,f,R)
    model.summary()
    