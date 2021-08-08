# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 17:25:52 2021

@author: arumugaraj
"""

import tensorflow as tf
from model import RNet, avg_NSR, avg_log_SNR
from read import read_data, print_data_samples, print_model_outputs
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# read the data
path = "D:\\project\\Dataset\\New folder\\*.*"
train_set_X, train_set_Y, val_set_X, val_set_Y, test_set_X, test_set_Y = read_data(path)

# print few samples from the training set
print_data_samples(train_set_X, train_set_Y)

# create an instance of the U-Net model
unet = RNet(input_size=208, input_channels=1, filters=32)
# unet.summary()

# set up checkpoint
checkpoint = ModelCheckpoint(filepath='./unet_checkpoints/', save_best_only=True, save_weights_only=True,monitor='mse', mode='max',  verbose=1)

# train the model
history = unet.fit(x=train_set_X, y=train_set_Y, validation_data=(val_set_X, val_set_Y), batch_size=2, epochs=6, callbacks=[checkpoint])


# load the weights of the model with best performance on the validation set
unet.load_weights('./unet_checkpoints/')

# check the performance of unet on the testing set
y_pred = unet.predict(test_set_X)
# print('average PSNRLoss of U-Net on the testing set: {}'.format(avg_log_SNR(test_set_Y, y_pred).numpy()))
print('average PSNRLoss of ResNet on the testing set: {}'.format(tf.math.reduce_mean(tf.image.psnr(test_set_Y, y_pred, 1.)).numpy()))
print('average SSIM of ResNet on the testing set: {}'.format(tf.math.reduce_mean(tf.image.ssim(test_set_Y, y_pred, 1.)).numpy()))

# check the performance of Input on the testing set
# print('average PSNRLoss of Input on the testing set: {}'.format(avg_log_SNR(test_set_Y, test_set_X).numpy()))
print('average PSNRLoss of Input on the testing set: {}'.format(tf.math.reduce_mean(tf.image.psnr(test_set_Y, test_set_X, 1.)).numpy()))
print('average SSIM of Input on the testing set: {}'.format(tf.math.reduce_mean(tf.image.ssim(test_set_Y, test_set_X, 1.)).numpy()))

# print few results of outputs of the U-Net on the training set
print_model_outputs(unet, train_set_X[:4], train_set_Y[:4], 'Training Data')

# print few results of outputs of the U-Net on the validation set
print_model_outputs(unet, val_set_X[:4], val_set_Y[:4], 'Validation Data')

# print few results of outputs of the model on the testing set
print_model_outputs(unet, test_set_X[:4], test_set_Y[:4], 'Testing Data')


