import numpy as N
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv1D, MaxPooling1D
import scipy.io as spio
import h5py
from keras.callbacks import ModelCheckpoint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Individual source model for the mPower TAPPING CNN

# fix random seed for reproducibility
seed = 7
N.random.seed(seed)


########################
# Load in all the data #
########################
f = h5py.File('CNN_xVal_X_Fold_LAST3.mat')

# Shape the training data
trainX = f["xTrain"]
trainY = f["yTrain"]

z, x, y = trainX.shape
B = N.reshape(trainX, [y*x*z])
trainX = N.reshape(B,[y, x, z], [y, x, z])

z, y = N.shape(trainY)
trainY = N.reshape(trainY, [y, z], [y, z])


# Shape the test data
testX = f["xTest"]
testY = f["yTest"]

z, x, y = testX.shape
B = N.reshape(testX, [y*x*z])
testX = N.reshape(B,[y, x, z], [y, x, z])

z, y = N.shape(testY)
testY = N.reshape(testY, [y, z], [y, z])

print('----------------')
print(trainX.shape)
print('----------------')
print(trainY.shape)
print('----------------')


####################
# Model Definition #
####################

def cnn_1d_all_signals():
    model = Sequential()

    # Layer 1
    model.add(Conv1D(8, 5, input_shape=(700, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.5))

    # Layer 2
    model.add(Conv1D(16, 5, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Layer 3
    model.add(Conv1D(32, 5, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.5))

    # Layer 4
    model.add(Conv1D(32, 5, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Layer 5
    model.add(Conv1D(64, 5, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.5))


    # Layer 6
    model.add(Conv1D(64, 5, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Layer 7
    model.add(Conv1D(128, 5, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.5))

    # Layer 8
    model.add(Conv1D(128, 5, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))

    # Layer 9
    model.add(Flatten())

    # Layer 4
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.5))

    # Layer 5
    model.add(Dense(300, activation='relu'))

    # Layer 5
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.5))

    # Layer 5
    model.add(Dense(100, activation='relu'))

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile Model
    model.compile(loss='binary_crossentropy',
              optimizer='adagrad',
              metrics=['accuracy'])
    print(model.summary())
    return model


# Set up the callbacks for saving the model with the best performance
callbacks = [ ModelCheckpoint('cnn_weights-best-last-3.txt',monitor='val_acc', save_best_only=True, verbose=2) ]

# Evaluate model with standardized dataset
model = cnn_1d_all_signals()
model.fit(trainX, trainY, validation_data=(testX,testY),batch_size=100, epochs=10000,verbose=1, shuffle=True, callbacks=callbacks)

# Load the best weights
model.load_weights('cnn_weights-best-last-3.txt')
yPredTEST = model.predict(testX)

#spio.savemat('CNN_xVal_Y_Fold_LAST10.mat', mdict={'yPred10': yPredTEST})
