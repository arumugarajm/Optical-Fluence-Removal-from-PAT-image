# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 17:25:52 2021

@author: arumugaraj
"""

import tensorflow as tf
from main import DenseUNet
from read import read_data, print_data_samples, print_model_outputs
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# read the data
path = "D:\\project\\Dataset\\New folder\\*.*"
X1,X2,X3,Z1,Z2,Z3 = read_data(path)

# print few samples from the training set
print_data_samples(X1, Z1)

# create the U-Net model
input_size = (256, 256, 1)
filters = 32
lr =  1e-4
model = DenseUNet(input_size, filters, lr)


# set up checkpoint
checkpoint = ModelCheckpoint(filepath='./model_checkpoints/', save_best_only=True, save_weights_only=True,monitor='mse', mode='min',  verbose=1)
# checkpoint = ModelCheckpoint(filepath='./model_checkpoints/', save_best_only=True, save_weights_only=True, monitor='val_avg_log_SNR', mode='max', verbose=1)

# train the model
history = model.fit(x=X1, y=Z1, validation_data=(X2, Z2), batch_size=15, epochs=60, callbacks=[checkpoint])

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
model.load_weights('./model_checkpoints/')

# check the performance of unet on the testing set
y_pred = model.predict(X3)
# print('average PSNRLoss of U-Net on the testing set: {}'.format(avg_log_SNR(test_set_Y, y_pred).numpy()))
print('average PSNRLoss of FD U-Net on the testing set: {}'.format(tf.math.reduce_mean(tf.image.psnr(Z3, y_pred, 1.)).numpy()))
print('average SSIM of FD U-Net on the testing set: {}'.format(tf.math.reduce_mean(tf.image.ssim(Z3, y_pred, 1.)).numpy()))

# check the performance of Input on the testing set
# print('average PSNRLoss of Input on the testing set: {}'.format(avg_log_SNR(test_set_Y, test_set_X).numpy()))
print('average PSNRLoss of Input on the testing set: {}'.format(tf.math.reduce_mean(tf.image.psnr(Z3, X3, 1.)).numpy()))
print('average SSIM of Input on the testing set: {}'.format(tf.math.reduce_mean(tf.image.ssim(Z3, X3, 1.)).numpy()))

# print few results of outputs of the U-Net on the training set
print_model_outputs(model, X1[:4], Z1[:4], 'Training Data')

# print few results of outputs of the U-Net on the validation set
print_model_outputs(model, X2[:4], Z2[:4], 'Validation Data')

# print few results of outputs of the model on the testing set
print_model_outputs(model, X3[:4], Z3[:4], 'Testing Data')

