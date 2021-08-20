# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 18:12:31 2021

@author: arumugaraj
"""


from tensorflow.keras.layers import Conv2D,Add, BatchNormalization,UpSampling2D, Activation, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# #batchnormalization+activation
# def Act(x, act=True): 
#     x = BatchNormalization()(x)
#     if act == True:
#         x = Activation("relu")(x)
#     return x
# #convolution block
# def CB(x, filters, kernel_size=(3, 3), padding="same", strides=1):
#     x = Act(x)
#     x = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
#     return x

# def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
#     x = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
#     x = CB(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
#     shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
#     shortcut = Act(shortcut, act=False)
    
#     output = Add()([x, shortcut])
#     return output
# #residual block
# def RB(x, filters, kernel_size=(3, 3), padding="same", strides=1):
#     res = CB(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
#     res = CB(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
#     shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
#     shortcut = Act(shortcut, act=False)
    
#     output = Add()([shortcut, res])
#     return output

# def upsample_concat_block(x, xskip):
#     u = UpSampling2D((2, 2))(x)
#     c = Concatenate()([u, xskip])
#     return c


# #residual unet
# def ResUnet(image_size,channels):
#     f = [16, 32, 64, 128, 256]
#     inputs = Input((image_size, image_size, channels))
    
#     # encoder
#     e0 = inputs
#     e1 = stem(e0, f[0])
#     e2 = RB(e1, f[1], strides=2)
#     e3 = RB(e2, f[2], strides=2)
#     e4 = RB(e3, f[3], strides=2)
#     e5 = RB(e4, f[4], strides=2)
    
#     # bridge
#     b0 = CB(e5, f[4], strides=1)
#     b1 = CB(b0, f[4], strides=1)
    
#     # decoder
#     u1 = upsample_concat_block(b1, e4)
#     d1 = RB(u1, f[4])
    
#     u2 = upsample_concat_block(d1, e3)
#     d2 = RB(u2, f[3])
    
#     u3 = upsample_concat_block(d2, e2)
#     d3 = RB(u3, f[2])
    
#     u4 = upsample_concat_block(d3, e1)
#     d4 = RB(u4, f[1])
    
#     outputs = Conv2D(1, (1, 1), padding="same", activation="relu")(d4)
#     model = Model(inputs, outputs)
#     model.compile(optimizer=Adam(lr = 1e-4), loss='mse', metrics=['mse'])
#     return model


# if __name__ == "__main__":
#     input_size = 208
#     channels = 1
#     model = ResUnet(input_size,channels)
#     model.summary()



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

def ResUnet(image_size,channels):
    f = [16, 32, 64, 128, 256]
    inputs = Input((image_size, image_size, 1))
    
    ## Encoder
    e0 = inputs #208, 208, 1
    # print(e0.shape)
    e1 = stem(e0, f[0]) #208, 208, 16
    # print(e1.shape)
    e2 = residual_block(e1, f[1], strides=2)#104, 104, 32
    # print(e2.shape)
    e3 = residual_block(e2, f[2], strides=2)#52, 52, 64
    # print(e3.shape)
    e4 = residual_block(e3, f[3], strides=2)#26, 26, 128
    # print(e4.shape)
    e5 = residual_block(e4, f[4], strides=2)#13, 13, 256
    # print(e5.shape)
    
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)#13, 13, 256
    # print(b0.shape)
    b1 = conv_block(b0, f[4], strides=1)#13, 13, 256
    # print(b1.shape)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e4) #26, 26, 384
    d1 = residual_block(u1, f[4])#26, 26, 256
    # print(u1.shape,d1.shape)
    
    u2 = upsample_concat_block(d1, e3)#52, 52, 320
    d2 = residual_block(u2, f[3])#52, 52, 128
    # print(u2.shape,d2.shape)
    
    u3 = upsample_concat_block(d2, e2)#104, 104, 160
    d3 = residual_block(u3, f[2])#104, 104, 64
    # print(u3.shape,d3.shape)
    
    u4 = upsample_concat_block(d3, e1)#208, 208, 80
    d4 = residual_block(u4, f[1])#208, 208, 32
    # print(u4.shape,d4.shape)
    
    outputs = Conv2D(1, (1, 1), padding="same", activation="relu")(d4)#208, 208, 1
    # print(outputs.shape)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(lr = 1e-4), loss='mse', metrics=['mse'])
    return model

if __name__ == "__main__":
    input_size = 208
    channels = 1
    model = ResUnet(input_size,channels)
    model.summary()

