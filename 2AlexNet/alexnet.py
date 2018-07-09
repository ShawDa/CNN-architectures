# -*- coding:utf-8 -*-

from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD


class AlexNet(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def alexnet(self):
        model = Sequential()
        # padding=same is optional
        model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(self.height, self.width, 3),padding='valid',activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

        model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

        model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu'))
        model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu'))
        model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))

        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(4096,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2,activation='softmax'))
        return model

    def train(self):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            '/data1/kaggle/dog_cat/train',
            batch_size=16,
            shuffle=True,
            target_size=(227, 227),
            class_mode='categorical')
        validation_generator = test_datagen.flow_from_directory(
            '/data1/kaggle/dog_cat/val',
            batch_size=16,
            target_size=(227, 227),
            shuffle=True,
            class_mode='categorical')

        model = self.alexnet()
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mse',
                        optimizer=sgd,
                        metrics=['accuracy'])
        history = model.fit_generator(train_generator,
                                        steps_per_epoch=2000,  # batch_size
                                        epochs=50,
                                        validation_data=validation_generator,
                                        validation_steps=500,  # batch_size
                                        verbose=1)
        model.save_weights('first_try.h5')


if __name__ == '__main__':
    alexnet = AlexNet(227, 227)
    model = alexnet.alexnet()
    model.summary()
    alexnet.train()
