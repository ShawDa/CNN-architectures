# -*- coding:utf-8 -*-

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import *
from keras.models import Sequential


def vgg16():
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=(32, 32, 3) ))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # layer2 32*32*64
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # layer3 16*16*64
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    # layer4 16*16*128
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # layer5 8*8*128
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    # layer6 8*8*256
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    # layer7 8*8*256
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # layer8 4*4*256
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    # layer9 4*4*512
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    # layer10 4*4*512
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # layer11 2*2*512
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    # layer12 2*2*512
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    # layer13 2*2*512
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    # layer14 1*1*512
    model.add(Flatten())
    model.add(Dense(512, ))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    # layer15 512
    model.add(Dense(512, ))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    # layer16 512
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Activation('softmax'))
    
    return model


# model = VGG16(weights=None, classes=100)
# model = VGG19(weights=None, classes=100)
model = vgg16()
model.summary()
