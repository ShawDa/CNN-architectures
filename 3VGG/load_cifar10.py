
import cv2
import numpy as np

from keras.datasets import cifar10
from keras import backend as K
from keras.utils import np_utils

nb_train_samples = 5000  # 50000 training samples
nb_valid_samples = 1000  # 10000 validation samples
num_classes = 10


def load_cifar10_data(img_rows, img_cols):

    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()
    print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape)

    # cv2.imwrite('before.png', X_train[0])

    # Resize trainging images
    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid[:nb_valid_samples,:,:,:]])
    else:
        X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]]).astype('float64')
        X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]]).astype('float64')

    # cv2.imwrite('bigger.png', X_train[0])

    while False:  # option
        # For Tensorflow
        # Switch RGB to BGR order
        X_train = X_train[:, :, :, ::-1]

        # Subtract ImageNet mean pixel
        X_train[:, :, :, 0] -= 103.939
        X_train[:, :, :, 1] -= 116.779
        X_train[:, :, :, 2] -= 123.68

        # Switch RGB to BGR order
        X_valid = X_valid[:, :, :, ::-1]

        # Subtract ImageNet mean pixel
        X_valid[:, :, :, 0] -= 103.939
        X_valid[:, :, :, 1] -= 116.779
        X_valid[:, :, :, 2] -= 123.68

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
    Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

    print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape)
    return X_train, Y_train, X_valid, Y_valid


if __name__ == '__main__':
    load_cifar10_data(224, 224)