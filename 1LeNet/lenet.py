# -*- coding:utf-8 -*-

from keras.models import Sequential
from keras.layers import *
from keras.datasets import mnist
import keras


class LeNet(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def lenet(self):
        model = Sequential()

        model.add(Conv2D(6, kernel_size=(5, 5), padding='valid', input_shape=(self.height, self.width, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        model.add(Conv2D(16, kernel_size=(5, 5), padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

        model.add(Flatten())
        model.add((Dense(84, activation='relu')))

        model.add(Dense(10, activation='softmax'))

        return model

    def train(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        model = self.lenet()
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        model.fit(np.concatenate([x_train, x_test]), np.concatenate([y_train, y_test]), batch_size=128,
                            epochs=100, verbose=1, validation_data=(x_test, y_test))

        score = model.evaluate(x_test, y_test, verbose=1)
        print(score)  # [3.308313189495493e-06, 1.0]


if __name__ == '__main__':
    lenet = LeNet(28, 28)
    model = lenet.lenet()
    model.summary()
    lenet.train()
