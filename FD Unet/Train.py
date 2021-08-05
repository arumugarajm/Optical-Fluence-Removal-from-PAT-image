# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 15:10:13 2021

@author: arumugaraj
"""

import tensorflow as tf
from utils import read_data, print_data_samples, print_model_outputs
from tensorflow.keras.callbacks import ModelCheckpoint
# from Dense import *
from main import DenseUNet

# read the data
# path = 'minideeplesion/*/*.png'
path = "D:\\project\\Dataset\\New folder\\*.*"
train_set_X, train_set_Y, val_set_X, val_set_Y, test_set_X, test_set_Y = read_data(path)

# print few samples from the training set
print_data_samples(train_set_X, train_set_Y)

# create an instance of the U-Net model
unet = DenseUNet(input_size=(208, 208, 1), filters=32, lr=1e-4)
unet.summary()


# set up checkpoint
checkpoint = ModelCheckpoint(filepath='./unet_checkpoints/', save_best_only=True, save_weights_only=True,monitor='val_avg_log_SNR', mode='max',  verbose=1)

# train the model
history = unet.fit(x=train_set_X, y=train_set_Y, validation_data=(val_set_X, val_set_Y), batch_size=2, epochs=10, callbacks=[checkpoint])

# # print the learning curve and the progress of average logSNR
# plt.plot(history.history['avg_log_SNR'])
# plt.title('avg_log_SNR')
# plt.show()

# plt.plot(history.history['loss'])
# plt.title('Learning Curve')
# plt.show()

# # plot metrics in a single plot
# plt.plot(history.history['mean_squared_error'])
# plt.plot(history.history['mean_absolute_error'])
# plt.plot(history.history['mean_absolute_percentage_error'])
# plt.plot(history.history['cosine_proximity'])
# plt.show()

# # plot loss during training
# plt.subplot(211)
# plt.title('Loss')
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()

# # plot mse during training
# plt.subplot(212)
# plt.title('Mean Squared Error')
# plt.plot(history.history['mean_squared_error'], label='train')
# plt.plot(history.history['val_mean_squared_error'], label='test')
# plt.legend()
# plt.show()

# load the weights of the model with best performance on the validation set
unet.load_weights('./unet_checkpoints/')

# check the performance of unet on the testing set
y_pred = unet.predict(test_set_X)
# print('average PSNRLoss of U-Net on the testing set: {}'.format(avg_log_SNR(test_set_Y, y_pred).numpy()))
print('average PSNRLoss of U-Net on the testing set: {}'.format(tf.math.reduce_mean(tf.image.psnr(test_set_Y, y_pred, 1.)).numpy()))
print('average SSIM of U-Net on the testing set: {}'.format(tf.math.reduce_mean(tf.image.ssim(test_set_Y, y_pred, 1.)).numpy()))


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
