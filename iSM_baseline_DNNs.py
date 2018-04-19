"""
DNN Classifier for feature based iSMs
J. Prince (c)
30/03/2018
john.prince@eng.ox.ac.uk
source ~/miniconda3/bin/activate deeplearn
"""
import numpy as N
import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling2D, GlobalAveragePooling1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.optimizers import SGD
import scipy.io as spio
from keras.models import Model
import h5py
from keras.callbacks import ModelCheckpoint
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import os
import usefulMethods as um
from keras.utils import plot_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import nNetModelDefinitions as blMods

# ###################
# Start the main loop
# ###################

"""
matrices format
MSL_Deep_Learning_Features_Model_XX.mat

Each contains:
--> Training
     -> trainX_lasso
     -> trainY

--> Testing
     -> testX_lasso
     -> testY
"""

for modelNum in range(1, 15):
    print("Current Model: " + str(modelNum))
    print(modelNum)

    if modelNum == 1:
        modStr = str(modelNum)
        modelDataString = "MSL_Deep_Learning_Features_Model_" + modStr + ".mat"  # Specify Model, Repetition, and Fold of Data File

        # Load the data
        modelData = h5py.File(modelDataString)  # Load Model Data File
        print(modelData)
        featureMatrix = modelData["trainX_lasso"]
        Y = modelData["trainY"]

        mainTestX = modelData["testX_lasso"]
        mainTestY = modelData["testY"]

        nSubs = Y.shape[1]
        print("There are " + str(nSubs) + " participants")
        nFolds = 10

        for rep in range(1, 2):

            # Need to Balance the groups
            balancedIndices = um.balance_data(Y[0])
            featureMatrix = featureMatrix[:, balancedIndices]
            Y = Y[:, balancedIndices]
            nBalSubs = len(balancedIndices)

            idx = um.cross_val_inds(nBalSubs, nFolds)

            for fold in range(1, 11):
                print("Rep: " + str(rep) + " Fold: " + str(fold) + " Model: " + str(modelNum))
                trainIdx, valIdx = um.kfold_train_test_idx(idx, fold)

                # Training Data
                featureTrain = featureMatrix[:, trainIdx]
                yTrain = Y[:, trainIdx]

                # Validation Data
                featureValidate = featureMatrix[:, valIdx]
                yVal = Y[:, valIdx]

                # Test Data
                featureTest = mainTestX
                yTest = mainTestY

                # Implement the model
                model_feature = blMods.feature_model_dnn(featureMatrix)

                # Reshape the TRAINING matrices
                featureX, featureY = featureTrain.shape
                B = N.reshape(featureTrain, [featureY * featureX])
                featureTrain = N.reshape(B, [featureY, featureX], [featureY, featureX])
                featureTrain = N.expand_dims(featureTrain, axis=2)

                # Reshape the VALIDATION matrices
                featureX, featureY = featureValidate.shape
                B = N.reshape(featureValidate, [featureY * featureX])
                featureValidate = N.reshape(B, [featureY, featureX], [featureY, featureX])
                featureValidate = N.expand_dims(featureValidate, axis=2)

                # Reshape the TEST matrices
                featureX, featureY = featureTest.shape
                B = N.reshape(featureTest, [featureY * featureX])
                featureTest = N.reshape(B, [featureY, featureX], [featureY, featureX])
                featureTest = N.expand_dims(featureTest, axis=2)

                # Reshape the TRAINING response vector
                z, y = N.shape(yTrain)
                yTrain = N.reshape(yTrain, [y, z], [y, z])

                # Reshape the VALIDATION response vector
                z, y = N.shape(yVal)
                yVal = N.reshape(yVal, [y, z], [y, z])

                # Reshape the TEST response vector
                z, y = N.shape(yTest)
                yTest = N.reshape(yTest, [y, z], [y, z])

                # Now can fit the model
                # Set up the callbacks for saving the model with the best performance
                callbacks = [
                    ModelCheckpoint('model1_TESTWeights.txt', monitor='loss', save_best_only=True, verbose=2)]

                history = model_feature.fit(featureTrain, yTrain, validation_data=(featureValidate, yVal), batch_size=32,
                                            epochs=1000,
                                            verbose=1,
                                            callbacks=callbacks)

                # Load best weights, predict, and save
                model_feature.load_weights('model1_TESTWeights.txt')

                # Save the TRAINING results
                yPredTRAIN = model_feature.predict(featureTrain)
                tempResultsTrain = N.zeros((yTrain.shape[0], 2), dtype=N.float64)
                tempResultsTrain[:, 0] = yTrain.reshape(yTrain.shape[0])
                tempResultsTrain[:, 1] = yPredTRAIN.reshape(yTrain.shape[0])

                # Save the VALIDATION results
                yPredVAL = model_feature.predict(featureValidate)
                tempResultsVal = N.zeros((yVal.shape[0], 2), dtype=N.float64)
                tempResultsVal[:, 0] = yVal.reshape(yVal.shape[0])
                tempResultsVal[:, 1] = yPredVAL.reshape(yVal.shape[0])

                # Save the TESTING results
                yPredTEST = model_feature.predict(featureTest)
                tempResultsTest = N.zeros((yTest.shape[0], 2), dtype=N.float64)
                tempResultsTest[:, 0] = yTest.reshape(yTest.shape[0])
                tempResultsTest[:, 1] = yPredTEST.reshape(yTest.shape[0])

                if (rep == 1) & (fold == 1):
                    model_results_val = tempResultsVal
                    model_results_train = tempResultsTrain
                    model_results_test = tempResultsTest
                else:
                    model_results_val = N.concatenate([model_results_val, tempResultsVal])
                    model_results_train = N.concatenate([model_results_train, tempResultsTrain])
                    model_results_test = N.concatenate([model_results_test, tempResultsTest])

            spio.savemat('Model 1 DNN Feature Results Training', mdict={'predictionsVal': model_results_val,
                                                                      'predictionsTrain': model_results_train,
                                                                      'predictionsTest': model_results_test})

    elif modelNum == 2:
        modStr = str(modelNum)
        modelDataString = "MSL_Deep_Learning_Features_Model_" + modStr + ".mat"  # Specify Model, Repetition, and Fold of Data File

        # Load the data
        modelData = h5py.File(modelDataString)  # Load Model Data File
        print(modelData)
        featureMatrix = modelData["trainX_lasso"]
        Y = modelData["trainY"]

        mainTestX = modelData["testX_lasso"]
        mainTestY = modelData["testY"]

        nSubs = Y.shape[1]
        print("There are " + str(nSubs) + " participants")
        nFolds = 10

        for rep in range(1, 2):

            # Need to Balance the groups
            balancedIndices = um.balance_data(Y[0])
            featureMatrix = featureMatrix[:, balancedIndices]
            Y = Y[:, balancedIndices]
            nBalSubs = len(balancedIndices)

            idx = um.cross_val_inds(nBalSubs, nFolds)

            for fold in range(1, 11):
                print("Rep: " + str(rep) + " Fold: " + str(fold) + " Model: " + str(modelNum))
                trainIdx, valIdx = um.kfold_train_test_idx(idx, fold)

                # Training Data
                featureTrain = featureMatrix[:, trainIdx]
                yTrain = Y[:, trainIdx]

                # Validation Data
                featureValidate = featureMatrix[:, valIdx]
                yVal = Y[:, valIdx]

                # Test Data
                featureTest = mainTestX
                yTest = mainTestY

                # Implement the model
                model_feature = blMods.feature_model_dnn(featureMatrix)

                # Reshape the TRAINING matrices
                featureX, featureY = featureTrain.shape
                B = N.reshape(featureTrain, [featureY * featureX])
                featureTrain = N.reshape(B, [featureY, featureX], [featureY, featureX])
                featureTrain = N.expand_dims(featureTrain, axis=2)

                # Reshape the VALIDATION matrices
                featureX, featureY = featureValidate.shape
                B = N.reshape(featureValidate, [featureY * featureX])
                featureValidate = N.reshape(B, [featureY, featureX], [featureY, featureX])
                featureValidate = N.expand_dims(featureValidate, axis=2)

                # Reshape the TEST matrices
                featureX, featureY = featureTest.shape
                B = N.reshape(featureTest, [featureY * featureX])
                featureTest = N.reshape(B, [featureY, featureX], [featureY, featureX])
                featureTest = N.expand_dims(featureTest, axis=2)

                # Reshape the TRAINING response vector
                z, y = N.shape(yTrain)
                yTrain = N.reshape(yTrain, [y, z], [y, z])

                # Reshape the VALIDATION response vector
                z, y = N.shape(yVal)
                yVal = N.reshape(yVal, [y, z], [y, z])

                # Reshape the TEST response vector
                z, y = N.shape(yTest)
                yTest = N.reshape(yTest, [y, z], [y, z])

                # Now can fit the model
                # Set up the callbacks for saving the model with the best performance
                callbacks = [
                    ModelCheckpoint('model1_TESTWeights.txt', monitor='loss', save_best_only=True, verbose=2)]

                history = model_feature.fit(featureTrain, yTrain, validation_data=(featureValidate, yVal),
                                            batch_size=32,
                                            epochs=1000,
                                            verbose=1,
                                            callbacks=callbacks)

                # Load best weights, predict, and save
                model_feature.load_weights('model1_TESTWeights.txt')

                # Save the TRAINING results
                yPredTRAIN = model_feature.predict(featureTrain)
                tempResultsTrain = N.zeros((yTrain.shape[0], 2), dtype=N.float64)
                tempResultsTrain[:, 0] = yTrain.reshape(yTrain.shape[0])
                tempResultsTrain[:, 1] = yPredTRAIN.reshape(yTrain.shape[0])

                # Save the VALIDATION results
                yPredVAL = model_feature.predict(featureValidate)
                tempResultsVal = N.zeros((yVal.shape[0], 2), dtype=N.float64)
                tempResultsVal[:, 0] = yVal.reshape(yVal.shape[0])
                tempResultsVal[:, 1] = yPredVAL.reshape(yVal.shape[0])

                # Save the TESTING results
                yPredTEST = model_feature.predict(featureTest)
                tempResultsTest = N.zeros((yTest.shape[0], 2), dtype=N.float64)
                tempResultsTest[:, 0] = yTest.reshape(yTest.shape[0])
                tempResultsTest[:, 1] = yPredTEST.reshape(yTest.shape[0])

                if (rep == 1) & (fold == 1):
                    model_results_val = tempResultsVal
                    model_results_train = tempResultsTrain
                    model_results_test = tempResultsTest
                else:
                    model_results_val = N.concatenate([model_results_val, tempResultsVal])
                    model_results_train = N.concatenate([model_results_train, tempResultsTrain])
                    model_results_test = N.concatenate([model_results_test, tempResultsTest])

            spio.savemat('Model 2 DNN Feature Results Training', mdict={'predictionsVal': model_results_val,
                                                               'predictionsTrain': model_results_train,
                                                               'predictionsTest': model_results_test})

    elif modelNum == 4:
        modStr = str(modelNum)
        modelDataString = "MSL_Deep_Learning_Features_Model_" + modStr + ".mat"  # Specify Model, Repetition, and Fold of Data File

        # Load the data
        modelData = h5py.File(modelDataString)  # Load Model Data File
        print(modelData)
        featureMatrix = modelData["trainX_lasso"]
        Y = modelData["trainY"]

        mainTestX = modelData["testX_lasso"]
        mainTestY = modelData["testY"]

        nSubs = Y.shape[1]
        print("There are " + str(nSubs) + " participants")
        nFolds = 10

        for rep in range(1, 2):

            # Need to Balance the groups
            balancedIndices = um.balance_data(Y[0])
            featureMatrix = featureMatrix[:, balancedIndices]
            Y = Y[:, balancedIndices]
            nBalSubs = len(balancedIndices)

            idx = um.cross_val_inds(nBalSubs, nFolds)

            for fold in range(1, 11):
                print("Rep: " + str(rep) + " Fold: " + str(fold) + " Model: " + str(modelNum))
                trainIdx, valIdx = um.kfold_train_test_idx(idx, fold)

                # Training Data
                featureTrain = featureMatrix[:, trainIdx]
                yTrain = Y[:, trainIdx]

                # Validation Data
                featureValidate = featureMatrix[:, valIdx]
                yVal = Y[:, valIdx]

                # Test Data
                featureTest = mainTestX
                yTest = mainTestY

                # Implement the model
                model_feature = blMods.feature_model_dnn(featureMatrix)

                # Reshape the TRAINING matrices
                featureX, featureY = featureTrain.shape
                B = N.reshape(featureTrain, [featureY * featureX])
                featureTrain = N.reshape(B, [featureY, featureX], [featureY, featureX])
                featureTrain = N.expand_dims(featureTrain, axis=2)

                # Reshape the VALIDATION matrices
                featureX, featureY = featureValidate.shape
                B = N.reshape(featureValidate, [featureY * featureX])
                featureValidate = N.reshape(B, [featureY, featureX], [featureY, featureX])
                featureValidate = N.expand_dims(featureValidate, axis=2)

                # Reshape the TEST matrices
                featureX, featureY = featureTest.shape
                B = N.reshape(featureTest, [featureY * featureX])
                featureTest = N.reshape(B, [featureY, featureX], [featureY, featureX])
                featureTest = N.expand_dims(featureTest, axis=2)

                # Reshape the TRAINING response vector
                z, y = N.shape(yTrain)
                yTrain = N.reshape(yTrain, [y, z], [y, z])

                # Reshape the VALIDATION response vector
                z, y = N.shape(yVal)
                yVal = N.reshape(yVal, [y, z], [y, z])

                # Reshape the TEST response vector
                z, y = N.shape(yTest)
                yTest = N.reshape(yTest, [y, z], [y, z])

                # Now can fit the model
                # Set up the callbacks for saving the model with the best performance
                callbacks = [
                    ModelCheckpoint('model1_TESTWeights.txt', monitor='loss', save_best_only=True, verbose=2)]

                history = model_feature.fit(featureTrain, yTrain, validation_data=(featureValidate, yVal),
                                            batch_size=32,
                                            epochs=1000,
                                            verbose=1,
                                            callbacks=callbacks)

                # Load best weights, predict, and save
                model_feature.load_weights('model1_TESTWeights.txt')

                # Save the TRAINING results
                yPredTRAIN = model_feature.predict(featureTrain)
                tempResultsTrain = N.zeros((yTrain.shape[0], 2), dtype=N.float64)
                tempResultsTrain[:, 0] = yTrain.reshape(yTrain.shape[0])
                tempResultsTrain[:, 1] = yPredTRAIN.reshape(yTrain.shape[0])

                # Save the VALIDATION results
                yPredVAL = model_feature.predict(featureValidate)
                tempResultsVal = N.zeros((yVal.shape[0], 2), dtype=N.float64)
                tempResultsVal[:, 0] = yVal.reshape(yVal.shape[0])
                tempResultsVal[:, 1] = yPredVAL.reshape(yVal.shape[0])

                # Save the TESTING results
                yPredTEST = model_feature.predict(featureTest)
                tempResultsTest = N.zeros((yTest.shape[0], 2), dtype=N.float64)
                tempResultsTest[:, 0] = yTest.reshape(yTest.shape[0])
                tempResultsTest[:, 1] = yPredTEST.reshape(yTest.shape[0])

                if (rep == 1) & (fold == 1):
                    model_results_val = tempResultsVal
                    model_results_train = tempResultsTrain
                    model_results_test = tempResultsTest
                else:
                    model_results_val = N.concatenate([model_results_val, tempResultsVal])
                    model_results_train = N.concatenate([model_results_train, tempResultsTrain])
                    model_results_test = N.concatenate([model_results_test, tempResultsTest])

            spio.savemat('Model 4 DNN Feature Results Training', mdict={'predictionsVal': model_results_val,
                                                               'predictionsTrain': model_results_train,
                                                               'predictionsTest': model_results_test})

    elif modelNum == 8:
        modStr = str(modelNum)
        modelDataString = "MSL_Deep_Learning_Features_Model_" + modStr + ".mat"  # Specify Model, Repetition, and Fold of Data File

        # Load the data
        modelData = h5py.File(modelDataString)  # Load Model Data File
        print(modelData)
        featureMatrix = modelData["trainX_lasso"]
        Y = modelData["trainY"]

        mainTestX = modelData["testX_lasso"]
        mainTestY = modelData["testY"]

        nSubs = Y.shape[1]
        print("There are " + str(nSubs) + " participants")
        nFolds = 10

        for rep in range(1, 3):

            # Need to Balance the groups
            balancedIndices = um.balance_data(Y[0])
            featureMatrix = featureMatrix[:, balancedIndices]
            Y = Y[:, balancedIndices]
            nBalSubs = len(balancedIndices)

            idx = um.cross_val_inds(nBalSubs, nFolds)

            for fold in range(1, 11):
                print("Rep: " + str(rep) + " Fold: " + str(fold) + " Model: " + str(modelNum))
                trainIdx, valIdx = um.kfold_train_test_idx(idx, fold)

                # Training Data
                featureTrain = featureMatrix[:, trainIdx]
                yTrain = Y[:, trainIdx]

                # Validation Data
                featureValidate = featureMatrix[:, valIdx]
                yVal = Y[:, valIdx]

                # Test Data
                featureTest = mainTestX
                yTest = mainTestY

                # Implement the model
                model_feature = blMods.feature_model_dnn(featureMatrix)

                # Reshape the TRAINING matrices
                featureX, featureY = featureTrain.shape
                B = N.reshape(featureTrain, [featureY * featureX])
                featureTrain = N.reshape(B, [featureY, featureX], [featureY, featureX])
                featureTrain = N.expand_dims(featureTrain, axis=2)

                # Reshape the VALIDATION matrices
                featureX, featureY = featureValidate.shape
                B = N.reshape(featureValidate, [featureY * featureX])
                featureValidate = N.reshape(B, [featureY, featureX], [featureY, featureX])
                featureValidate = N.expand_dims(featureValidate, axis=2)

                # Reshape the TEST matrices
                featureX, featureY = featureTest.shape
                B = N.reshape(featureTest, [featureY * featureX])
                featureTest = N.reshape(B, [featureY, featureX], [featureY, featureX])
                featureTest = N.expand_dims(featureTest, axis=2)

                # Reshape the TRAINING response vector
                z, y = N.shape(yTrain)
                yTrain = N.reshape(yTrain, [y, z], [y, z])

                # Reshape the VALIDATION response vector
                z, y = N.shape(yVal)
                yVal = N.reshape(yVal, [y, z], [y, z])

                # Reshape the TEST response vector
                z, y = N.shape(yTest)
                yTest = N.reshape(yTest, [y, z], [y, z])

                # Now can fit the model
                # Set up the callbacks for saving the model with the best performance
                callbacks = [
                    ModelCheckpoint('model1_TESTWeights.txt', monitor='loss', save_best_only=True, verbose=2)]

                history = model_feature.fit(featureTrain, yTrain, validation_data=(featureValidate, yVal),
                                            batch_size=32,
                                            epochs=1000,
                                            verbose=1,
                                            callbacks=callbacks)

                # Load best weights, predict, and save
                model_feature.load_weights('model1_TESTWeights.txt')

                # Save the TRAINING results
                yPredTRAIN = model_feature.predict(featureTrain)
                tempResultsTrain = N.zeros((yTrain.shape[0], 2), dtype=N.float64)
                tempResultsTrain[:, 0] = yTrain.reshape(yTrain.shape[0])
                tempResultsTrain[:, 1] = yPredTRAIN.reshape(yTrain.shape[0])

                # Save the VALIDATION results
                yPredVAL = model_feature.predict(featureValidate)
                tempResultsVal = N.zeros((yVal.shape[0], 2), dtype=N.float64)
                tempResultsVal[:, 0] = yVal.reshape(yVal.shape[0])
                tempResultsVal[:, 1] = yPredVAL.reshape(yVal.shape[0])

                # Save the TESTING results
                yPredTEST = model_feature.predict(featureTest)
                tempResultsTest = N.zeros((yTest.shape[0], 2), dtype=N.float64)
                tempResultsTest[:, 0] = yTest.reshape(yTest.shape[0])
                tempResultsTest[:, 1] = yPredTEST.reshape(yTest.shape[0])

                if (rep == 1) & (fold == 1):
                    model_results_val = tempResultsVal
                    model_results_train = tempResultsTrain
                    model_results_test = tempResultsTest
                else:
                    model_results_val = N.concatenate([model_results_val, tempResultsVal])
                    model_results_train = N.concatenate([model_results_train, tempResultsTrain])
                    model_results_test = N.concatenate([model_results_test, tempResultsTest])

            spio.savemat('Model 8 DNN Feature Results Training', mdict={'predictionsVal': model_results_val,
                                                               'predictionsTrain': model_results_train,
                                                               'predictionsTest': model_results_test})