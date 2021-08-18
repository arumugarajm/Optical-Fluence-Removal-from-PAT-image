# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 17:25:52 2021

@author: arumugaraj
"""

from Ynet import Y_net,avg_NSR, avg_log_SNR
from read import read_data,print_data_samples,print_model_outputs
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint



# read the data
path = "D:\\project\\Dataset\\New folder (2)\\New folder\\*.*"
X1,X2,X3,Y1,Y2,Y3,Z1,Z2,Z3 = read_data(path)

# print few samples from the training set
print_data_samples(X1, Y1, Z1)


# create an instance of the U-Net model
unet = Y_net((256, 256, 1),32,(256,512,1))

# set up checkpoint
checkpoint = ModelCheckpoint(filepath='./unet_checkpoints/', save_best_only=True, save_weights_only=True, monitor="val_loss", mode='max', verbose=1)

# train the model
# history = unet.fit(x=train_set_X, y=train_set_Y, validation_data=(val_set_X, val_set_Y), batch_size=2, epochs=1)
history = unet.fit(
	x=[X1,Y1], y=Z1,
	validation_data=([X2, Y2], Z2),
	epochs=1, batch_size=2,callbacks=[checkpoint])

# load the weights of the model with best performance on the validation set
unet.load_weights('./unet_checkpoints/')

# check the performance of unet on the testing set
y_pred = unet.predict([X3,Y3])

# # check the performance of unet on the testing set
# y_pred = unet.predict([val_set_X,val_set_X1])

# # print few results of outputs of the model on the testing set
# print_model_outputs(unet, test_set_X[:4], test_set_Y[:4], 'Testing Data')






