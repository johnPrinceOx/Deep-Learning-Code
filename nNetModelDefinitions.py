"""
Model Definitions for CNN-MSL via mulit-task learning
J. Prince (c)
08/03/2018
john.prince@eng.ox.ac.uk
"""

import numpy as N
import pandas
import keras
from keras import optimizers, regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras.layers import Embedding
from keras.layers import Conv1D, LSTM, MaxPooling1D
from keras.optimizers import SGD
import scipy.io as spio
from keras.models import Model
import h5py
from keras.layers.merge import concatenate
import os
from keras.utils import plot_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt

""" Specify the structures of each of the three raw CNNs"""
# ############################################################
# Specify the structures of each of the three CNNs and the DNN
##############################################################


def tapping_cnn(inputsTapping, tapX, tapZ):
    """ This method implements the raw tapping CNN and returns the features"""
    l2_lambda = 0.00005

    # Layer 1
    conv1 = Conv1D(128, 400, input_shape=(tapX, tapZ), padding="valid", kernel_regularizer=regularizers.l2(5e-4))(inputsTapping)
    batch1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(batch1)
    pool1 = MaxPooling1D(4)(act1)
    drop1 = Dropout(0.5)(pool1)

    # Layer 2
    conv2 = Conv1D(128, 10, padding="valid", kernel_regularizer=regularizers.l2(5e-4), strides=2)(drop1)
    batch2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(batch2)
    drop2 = Dropout(0.5)(act2)

    # Layer 3
    conv3 = Conv1D(128, 10, padding="valid", kernel_regularizer=regularizers.l2(1e-4), strides=2)(drop2)
    batch3 = BatchNormalization()(conv3)
    act3 = Activation('relu')(batch3)
    drop3 = Dropout(0.5)(act3)  # Model 8, Accuracy~70%

    # Layer 4
    conv4 = Conv1D(128, 10, padding="valid", kernel_regularizer=regularizers.l2(1e-4), strides=2)(drop3)
    batch4 = BatchNormalization()(conv4)
    act4 = Activation('relu')(batch4)

    # Layer 5
    conv5 = Conv1D(256, 10, padding="valid", kernel_regularizer=regularizers.l2(1e-4), strides=2)(act4)
    batch5 = BatchNormalization()(conv5)
    act5 = Activation('relu')(batch5)
    drop5 = Dropout(0.5)(act5)

    # Layer 6
    conv6 = Conv1D(256, 5, padding="valid", kernel_regularizer=regularizers.l2(1e-3))(drop5)
    batch6 = BatchNormalization()(conv6)
    act6 = Activation('relu')(batch6)
    drop6 = Dropout(0.5)(act6)

    # Layer 7
    conv7 = Conv1D(256, 5, padding="valid", kernel_regularizer=regularizers.l2(5e-3))(drop6)
    batch7 = BatchNormalization()(conv7)
    act7 = Activation('relu')(batch7)
    drop7 = Dropout(0.5)(act7)

    # Layer 8
    flatTap = Flatten()(drop7)
    return flatTap


def walking_cnn(inputsWalking, gaitX, gaitZ):
    """ This method implements the raw walking CNN and returns the features"""
    l2_lambda = 0.00005

    # Layer 1
    conv1 = Conv1D(64, 400, input_shape=(gaitX, gaitZ), padding="valid", strides=5)(inputsWalking)
    batch1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(batch1)
    pool1 = MaxPooling1D(2)(act1)
    drop1 = Dropout(0.5)(pool1)

    # Layer 2
    conv2 = Conv1D(256, 10, padding="valid", strides=2)(pool1)
    batch2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(batch2)
    drop2 = Dropout(0.5)(act2)

    # Layer 3
    conv3 = Conv1D(256, 10, padding="valid", strides=2)(act2)
    batch3 = BatchNormalization()(conv3)
    act3 = Activation('relu')(batch3)
    drop3 = Dropout(0.3)(act3)
    """
    # Layer 4
    conv4 = Conv1D(256, 10, padding="valid")(act3)
    batch4 = BatchNormalization()(conv4)
    act4 = Activation('relu')(batch4)
    #drop4 = Dropout(0.5)(act4)  # Gives ~63-70%

    # Layer 5
    conv5 = Conv1D(512, 10, padding="valid")(act4)
    batch5 = BatchNormalization()(conv5)
    act5 = Activation('relu')(batch5)
    pool5 = MaxPooling1D(2)(act5)
    drop5 = Dropout(0.5)(pool5)

    
    # Layer 6
    conv6 = Conv1D(64, 2, padding="valid")(drop5)
    batch6 = BatchNormalization()(conv6)
    act6 = Activation('relu')(batch6)
    #drop6 = Dropout(0.5)(act6)

    # Layer 7
    conv7 = Conv1D(128, 5, padding="valid")(act6)
    batch7 = BatchNormalization()(conv7)
    act7 = Activation('relu')(batch7)
    pool7 = MaxPooling1D(2)(act7)
    #drop7 = Dropout(0.5)(pool7)

    # Layer 8
    conv8 = Conv1D(128, 5, padding="valid")(pool7)
    batch8 = BatchNormalization()(conv8)
    act8 = Activation('relu')(batch8)
    pool8 = MaxPooling1D(2)(act8)
    drop8 = Dropout(0.1)(pool8)
    """
    # Layer 9
    flatGait = Flatten()(drop3)
    return flatGait


def voice_cnn(inputsVoice, voiceX):
    """ This method implements the raw voice CNN and returns the features"""
    l2_lambda = 5e-4

    # Layer 1
    conv1 = Conv1D(128, 46000, input_shape=(voiceX, 1), padding="valid", strides=10000)(inputsVoice)
    batch1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(batch1)
    #pool1 = MaxPooling1D(2)(act1)
    drop1 = Dropout(0.5)(act1)

    # Layer 2 - Purely an extension/exaggerator layer to add further abstraction to the lower frequency components
    conv2 = Conv1D(256, 10, padding="valid", strides=1, kernel_regularizer=regularizers.l2(l2_lambda))(drop1)
    batch2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(batch2)
    drop2 = Dropout(0.5)(act2)

    """
    # Layer 3
    conv3 = Conv1D(512, 10, padding="valid", strides=2)(drop2)
    batch3 = BatchNormalization()(conv3)
    act3 = Activation('relu')(batch3)
    drop3 = Dropout(0.5)(act3)
    
    # Layer 4
    conv4 = Conv1D(256, 5, padding="valid", strides=2, kernel_regularizer=regularizers.l2(5e-5))(drop3)
    batch4 = BatchNormalization()(conv4)
    act4 = Activation('relu')(batch4)
    drop4 = Dropout(0.5)(act4)

    # Layer 5
    conv5 = Conv1D(512, 5, padding="valid", kernel_regularizer=regularizers.l2(5e-5))(drop4)
    batch5 = BatchNormalization()(conv5)
    act5 = Activation('relu')(batch5)
    pool5 = MaxPooling1D(2)(act5)
    drop5 = Dropout(0.5)(pool5)

    # LSTM to account for amplitude variation at the large time scale
    #lstm1 = LSTM(500, activation='relu')

    
    # Layer 6
    conv6 = Conv1D(128, 2, padding="valid", kernel_regularizer=regularizers.l2(l2_lambda))(drop5)
    batch6 = BatchNormalization()(conv6)
    act6 = Activation('relu')(batch6)
    pool6 = MaxPooling1D(2)(act6)
    drop6 = Dropout(0.5)(pool6)

    # Layer 7
    conv7 = Conv1D(128, 2, padding="valid")(drop6)
    batch7 = BatchNormalization()(conv7)
    act7 = Activation('relu')(batch7)
    pool7 = MaxPooling1D(2)(act7)
    drop7 = Dropout(0.5)(pool7)

    # Layer 8
    conv8 = Conv1D(64, 2, padding="valid", kernel_regularizer=regularizers.l2(l2_lambda))(drop7)
    batch8 = BatchNormalization()(conv8)
    act8 = Activation('relu')(batch8)
    pool8 = MaxPooling1D(2)(act8)
    drop8 = Dropout(0.5)(pool8)
    """

    # Layer 9
    flatVoice = Flatten()(drop2)
    return flatVoice


def baseline_dnn(merged):
    dense1 = Dense(300, activation='relu')(merged)
    dnnDrop1 = Dropout(0.5)(dense1)
    dense2 = Dense(400, activation='relu')(dnnDrop1)
    dnnDrop2 = Dropout(0.5)(dense2)
    dense3 = Dense(100, activation='relu')(dnnDrop2)
    dnnDrop3 = Dropout(0.5)(dense3)
    dense4 = Dense(64, activation='relu')(dnnDrop3)
    dnnDrop4 = Dropout(0.5)(dense4)
    finalOutputs = Dense(1, activation='sigmoid')(dnnDrop4)
    return finalOutputs


def feature_dnn(merged):
    l2_lambda = 0.00005

    dense1 = Dense(200, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(merged) #300
    dnnDrop1 = Dropout(0.5)(dense1)
    dense2 = Dense(300, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(dnnDrop1) #400
    dnnDrop2 = Dropout(0.5)(dense2)
    dense3 = Dense(50, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(dnnDrop2) #100
    dnnDrop3 = Dropout(0.5)(dense3)
    dense4 = Dense(32, activation='relu')(dnnDrop3) # 64
    dnnDrop4 = Dropout(0.5)(dense4)
    finalOutputs = Dense(1, activation='sigmoid')(dnnDrop4)
    return finalOutputs


def feature_dnn2(merged):
    l2_lambda = 0.0005

    dense1 = Dense(300, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(merged) #300
    dnnDrop1 = Dropout(0.5)(dense1)
    dense2 = Dense(50, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(dnnDrop1) #400
    dnnDrop2 = Dropout(0.5)(dense2)
    dense3 = Dense(50, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(dnnDrop2) #100
    dnnDrop3 = Dropout(0.5)(dense3)
    dense4 = Dense(32, activation='relu')(dnnDrop3) # 64
    dnnDrop4 = Dropout(0.5)(dense4)
    finalOutputs = Dense(1, activation='sigmoid')(dnnDrop4)
    return finalOutputs


def tapping_cnn_dnn(merged):
    l2_lambda = 0.0005

    dense1 = Dense(300, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(merged) #300
    dnnDrop1 = Dropout(0.5)(dense1)
    dense2 = Dense(50, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(dnnDrop1) #400
    dnnDrop2 = Dropout(0.5)(dense2)
    dense3 = Dense(50, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(dnnDrop2) #100
    dnnDrop3 = Dropout(0.5)(dense3)
    dense4 = Dense(32, activation='relu')(dnnDrop3) # 64
    dnnDrop4 = Dropout(0.5)(dense4)
    finalOutputs = Dense(1, activation='sigmoid')(dnnDrop4)
    return finalOutputs


def voice_cnn_dnn(merged):
    l2_lambda = 0.0005

    dense1 = Dense(300, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(merged) #300
    dnnDrop1 = Dropout(0.5)(dense1)
    dense2 = Dense(50, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(dnnDrop1) #400
    dnnDrop2 = Dropout(0.5)(dense2)
    dense3 = Dense(50, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(dnnDrop2) #100
    dnnDrop3 = Dropout(0.5)(dense3)
    dense4 = Dense(32, activation='relu')(dnnDrop3) # 64
    dnnDrop4 = Dropout(0.5)(dense4)
    finalOutputs = Dense(1, activation='sigmoid')(dnnDrop4)
    return finalOutputs


def walking_cnn_dnn(merged):
    l2_lambda = 0.000005

    dense1 = Dense(50, activation='relu')(merged) #300
    dnnDrop1 = Dropout(0.5)(dense1)
    dense2 = Dense(50, activation='relu')(dnnDrop1) #400
    #dnnDrop2 = Dropout(0.5)(dense2)
    dense3 = Dense(50, activation='relu')(dense2) #100
    #dnnDrop3 = Dropout(0.5)(dense3)
    dense4 = Dense(32, activation='relu')(dense3) # 64
    dnnDrop4 = Dropout(0.4)(dense4)
    finalOutputs = Dense(1, activation='sigmoid')(dnnDrop4)
    return finalOutputs


# ##############################################################
# Code up each of the iSMs using the above CNNs
################################################################


def feature_model_dnn(featureMatrix):
    """ Model for target domain [0 0 0 1]"""

    # ####################
    # Reshape the matrices
    # ####################

    # memory data is only 2D
    featureMatrix = N.expand_dims(featureMatrix, axis=0)
    somethingZ, featureX, featureY = featureMatrix.shape

    # ########################
    # Build the Model Channels
    # ########################

    # Memory Channel
    inputsFeature = Input(shape=(featureX, 1), dtype='float32')
    flatFeature = Flatten()(inputsFeature)

    # Merge the Features
    merged = flatFeature

    # Put Features into DNN
    finalOutputs = feature_dnn(merged)

    # Create & Compile Model
    model = Model(inputs=[inputsFeature], outputs=finalOutputs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())
    return model


def model_1_cnn(memoryMatrix):
    """ Model for target domain [0 0 0 1]"""

    # ####################
    # Reshape the matrices
    # ####################

    # memory data is only 2D
    memoryMatrix = N.expand_dims(memoryMatrix, axis=0)
    somethingZ, memoryX, memoryY = memoryMatrix.shape

    # ########################
    # Build the Model Channels
    # ########################

    # Memory Channel
    inputsMemory = Input(shape=(memoryX, 1), dtype='float32')
    flatMemory = Flatten()(inputsMemory)

    # Merge the Features
    merged = flatMemory

    # Put Features into DNN
    finalOutputs = baseline_dnn(merged)

    # Create & Compile Model
    model = Model(inputs=[inputsMemory], outputs=finalOutputs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())
    return model


def model_2_cnn(voiceMatrix):
    """ Model for target domain [0 0 1 0]"""

    # ####################
    # Reshape the matrices
    # ####################

    # Voice data is only 2D
    voiceMatrix = N.expand_dims(voiceMatrix, axis=0)
    somethingZ, voiceX, voiceY = voiceMatrix.shape

    # ########################
    # Build the Model Channels
    # ########################

    # Voice Channel
    inputsVoice = Input(shape=(voiceX, 1))
    flatVoice = voice_cnn(inputsVoice, voiceX)

    # Merge the Features
    merged = flatVoice

    # Put Features into DNN
    finalOutputs = feature_dnn2(merged)

    # Create & Compile Model
    model = Model(inputs=[inputsVoice], outputs=finalOutputs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adagrad',
                  metrics=['accuracy'])

    print(model.summary())
    return model


def model_4_cnn(walkingMatrix):
    """ Model for target domain [0 1 0 0]"""

    # ####################
    # Reshape the matrices
    # ####################

    # For 3D Gait Data
    gaitZ, gaitX, gaitY = walkingMatrix.shape

    # ########################
    # Build the Model Channels
    # ########################

    # Gait Channel
    inputsWalking = Input(shape=(gaitX, gaitZ))
    flatGait = walking_cnn(inputsWalking, gaitX, gaitZ)

    # Merge the Features
    merged = flatGait

    # Put Features into DNN
    finalOutputs = walking_cnn_dnn(merged)

    # Create & Compile Model
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model = Model(inputs=[inputsWalking], outputs=finalOutputs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    #print(model.summary())
    return model


def model_8_cnn(tappingMatrix):
    """ Model for target domain [1 0 0 0]"""

    # ####################
    # Reshape the matrices
    # ####################

    # For 3D Tapping Data
    tapZ, tapX, tapY = tappingMatrix.shape

    # ########################
    # Build the Model Channels
    # ########################

    # Tapping CNN
    inputsTapping = Input(shape=(tapX, tapZ))
    flatTap = tapping_cnn(inputsTapping, tapX, tapZ)

    # Merge the Features
    merged = flatTap

    # Put Features into DNN
    finalOutputs = tapping_cnn_dnn(merged)

    # Create & Compile Model
    model = Model(inputs=[inputsTapping], outputs=finalOutputs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    #print(model.summary())
    return model
