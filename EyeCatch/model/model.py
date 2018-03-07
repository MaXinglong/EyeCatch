# #!usr/bin/env python3

"""
model 
1. ConvNet
2. U-Net -> ConvNet
"""

# from keras.layers import Input, Dense, Conv2D
# from keras.layers import Model

# def get_ConvNet_model():
#     inputs = Input(shape=(480, 640))

# for test
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as mobilenet_preprocess

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten

# import numpy as np

def get_model():
    resnet_model = ResNet50(weights='imagenet')

    x = resnet_model.get_layer('flatten_1').output
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation=None)(x)

    model = Model(input=resnet_model.input, output=predictions)

    for layer in resnet_model.layers:
        layer.trainable = False

    return model

def preprocess(X):
    X = preprocess_input(X)
    return X

def preprocess_mobilenet(X):
    X = preprocess_mobilenet(X)
    return X

def get_UNet():
    None
    
def get_MobileNet():
    mobilenet_model = MobileNet(weights='imagenet')

    x = mobilenet_model.get_layer('conv_preds').output
    x = Flatten(x)
    predictions = Dense(2, activation=2)(x)
    
    model = Model(input=mobilenet_model.input, output=predictions)

    # for layer in resnet_model.layers:
    #     layer.trainable = False
    return model
