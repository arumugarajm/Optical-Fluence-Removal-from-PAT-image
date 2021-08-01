import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
from skimage.transform import radon, iradon
import scipy.io



def read_data(path):
    i1=12
    i2=16

    training_images = []
    inverted_images = []
    

    for file_name in glob.glob(path): #+'/**/*.jpg', recursive=True):
        # print(file_name)
        mat=scipy.io.loadmat(file_name)
        a1=mat['Xim']
        b1=mat['Xi']
        a=a1/0.0438
        b=b1/0.0438
        image = np.asarray(a)
        invimage =np.asarray(b)
        training_images.append(image)
        inverted_images.append(invimage)
        
       
    
    training_images = np.asarray(training_images)
    inverted_images = np.asarray(inverted_images)
    

    return np.expand_dims(inverted_images[: i1], axis=-1).astype('float32'), np.expand_dims(training_images[: i1], axis=-1).astype('float32'), np.expand_dims(inverted_images[i1:i2], axis=-1).astype('float32'), np.expand_dims(training_images[i1:i2], axis=-1).astype('float32'), np.expand_dims(inverted_images[i2:], axis=-1).astype('float32'), np.expand_dims(training_images[i2:], axis=-1).astype('float32')


def print_data_samples(input_X, input_Y):
    """
    Prints 4 pair of images (Y, X), where:
    input_Y --- ground truth
    input_X --- noisy approximation of Y
    """
    plt.figure(figsize=(12., 6.))

    for i in range(1,5):
        plt.subplot(2, 4, i)
        plt.imshow(input_X[i,:,:,0], cmap='gray')
        plt.axis('off')
        plt.subplot(2, 4, i + 4)
        plt.imshow(input_Y[i,:,:,0], cmap='gray')
        plt.axis('off')

    plt.show()
    ar=input_X[i,:,:,0]
    # print(ar)

def print_model_outputs(model, input_X, input_Y, title):
    """
    Prints 4 triple of images (prediction, input_X, input_Y), where:
    input_Y --- ground truth
    input_X --- noisy approximation of Y
    prediction = model(input_X) --- cleaned X
    """
    prediction = model(input_X, training=False)

    fig = plt.figure(figsize=(9., 9.))

    for i in range(1,4):
        plt.subplot(3, 3, (i - 1) * 3 + 1)
        plt.imshow(prediction[i - 1,:,:,0], cmap='gray')
        plt.axis('off')
        if i == 1:
            plt.text(x=42, y=-4, s='Unet')
        plt.subplot(3, 3, (i - 1) * 3 + 2)
        plt.imshow(input_X[i - 1,:,:,0], cmap='gray')
        plt.axis('off')
        if i == 1:
            plt.text(x=56, y=-4, s='Input')
        plt.subplot(3, 3, (i - 1) * 3 + 3)
        plt.imshow(input_Y[i - 1,:,:,0], cmap='gray')
        plt.axis('off')
        if i == 1:
            plt.text(x=36, y=-4, s='Ground Truth')
    
    fig.suptitle(title, fontsize=16, y=0.94)
    plt.show()
