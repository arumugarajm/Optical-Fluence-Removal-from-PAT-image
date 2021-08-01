# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 15:07:28 2021

@author: arumugaraj
"""
import tensorflow as tf
def avg_NSR(Y_true, Y_pred):
    signal = tf.norm(tf.reshape(Y_true, [tf.shape(Y_true)[0], -1]), axis=1)
    noise = tf.norm(tf.reshape(Y_true - Y_pred, [tf.shape(Y_true)[0], -1]), axis=1)

    return tf.math.reduce_mean(noise / signal)

def avg_log_SNR(Y_true, Y_pred):
    signal = tf.norm(tf.reshape(Y_true, [tf.shape(Y_true)[0], -1]), axis=1)
    noise = tf.norm(tf.reshape(Y_true - Y_pred, [tf.shape(Y_true)[0], -1]), axis=1)

    return tf.math.reduce_mean(20. * tf.math.log(signal / noise) / tf.math.log(10.0))


from keras.layers import Conv2D,Input,MaxPooling2D,Conv2DTranspose,concatenate,ReLU
from Dense import BatchActivate,DenseBlock
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
def DenseUNet(input_size, start_neurons, lr):

    X = Input(input_size)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(X)
    conv1 = BatchActivate(conv1)
    conv1 = DenseBlock(start_neurons * 2, conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = DenseBlock(start_neurons * 4, pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = DenseBlock(start_neurons * 8, pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)


    conv4 = DenseBlock(start_neurons * 16, pool3)
    pool4 = MaxPooling2D((2, 2))(conv4)
    
    
    conv5 = DenseBlock(start_neurons * 32, pool4)


    deconv4 = Conv2DTranspose(start_neurons * 32, (3, 3), strides=(2, 2), padding="same")(conv5)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Conv2D(start_neurons * 8, (1, 1), activation=None, padding="same")(uconv4)
    uconv4 = BatchActivate(uconv4)
    uconv4 = DenseBlock(start_neurons * 16, uconv4)


    deconv3 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Conv2D(start_neurons * 4, (1, 1), activation=None, padding="same")(uconv3)
    uconv3 = BatchActivate(uconv3)
    uconv3 = DenseBlock(start_neurons * 8, uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D(start_neurons * 2, (1, 1), activation=None, padding="same")(uconv2)
    uconv2 = BatchActivate(uconv2)
    uconv2 = DenseBlock(start_neurons * 4, uconv2)
    
    
    deconv1 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D(start_neurons * 1, (1, 1), activation=None, padding="same")(uconv1)
    uconv1 = BatchActivate(uconv1)
    uconv1 = DenseBlock(start_neurons * 2, uconv1)

    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output = ReLU()(output_layer_noActi)
    # output = Activation('sigmoid')(output_layer_noActi)
    # output = output + X

    model = Model(inputs=X, outputs=output)

    model.compile(optimizer=Adam(lr=lr), loss=avg_NSR, metrics=[avg_log_SNR])

    return model