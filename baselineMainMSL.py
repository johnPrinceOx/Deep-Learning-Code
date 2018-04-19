"""
Main implementation of CNN-MSL via multi-task learning
J. Prince (c)
08/03/2018
john.prince@eng.ox.ac.uk
~/miniconda3/bin/activate deeplearn
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


for modelNum in range(2, 3):
    print("Current Model: " + str(modelNum))
    print(modelNum)

    if modelNum == 1:
        modStr = str(modelNum)
        modelDataString = "MSL_Deep_Learning_Model_" + modStr + ".mat"  # Specify Model, Repetition, and Fold of Data File

        # Load the data
        modelData = h5py.File(modelDataString)  # Load Model Data File
        memoryMatrix = modelData["memoryMatrix"]
        Y = modelData["Y"]

        nSubs = Y.shape[1]
        print("There are " + str(nSubs) + " participants")
        nFolds = 10

        for rep in range(1, 2):

            # Need to Balance the groups
            balancedIndices = um.balance_data(Y[0])
            print(memoryMatrix.shape)
            memoryMatrix = memoryMatrix[:, balancedIndices]
            print(memoryMatrix.shape)
            Y = Y[:, balancedIndices]
            nBalSubs = len(balancedIndices)

            idx = um.cross_val_inds(nBalSubs, nFolds)

            for fold in range(1, 11):
                print("Rep: " + str(rep) + " Fold: " + str(fold))
                trainIdx, testIdx = um.kfold_train_test_idx(idx, fold)

                # Training Data
                memoryTrain = memoryMatrix[:, trainIdx]
                yTrain = Y[:, trainIdx]
                print(memoryTrain.shape)

                # Testing Data
                memoryTest = memoryMatrix[:, testIdx]
                yTest = Y[:, testIdx]
                print(memoryTest.shape)

                # Implement the model
                model_1 = blMods.model_1_cnn(memoryMatrix)

                # Reshape the TRAINING matrices
                # Walking data is 3D
                # For 3D Gait Data
                memoryX, memoryY = memoryTrain.shape
                B = N.reshape(memoryTrain, [memoryY * memoryX])
                memoryTrain = N.reshape(B, [memoryY, memoryX], [memoryY, memoryX])
                memoryTrain = N.expand_dims(memoryTrain, axis=2)

                # Reshape the TESTING matrices
                # Voice data is 2D
                memoryX, memoryY = memoryTest.shape
                B = N.reshape(memoryTest, [memoryY * memoryX])
                memoryTest = N.reshape(B, [memoryY, memoryX], [memoryY, memoryX])
                memoryTest = N.expand_dims(memoryTest, axis=2)

                # Reshape the TRAINING response vector
                z, y = N.shape(yTrain)
                yTrain = N.reshape(yTrain, [y, z], [y, z])

                # Reshape the TESTING response vector
                z, y = N.shape(yTest)
                yTest = N.reshape(yTest, [y, z], [y, z])

                # Now can fit the model
                # Set up the callbacks for saving the model with the best performance
                callbacks = [
                    ModelCheckpoint('model1_TESTWeights.txt', monitor='val_loss', save_best_only=True, verbose=0)]

                history = model_1.fit(memoryTrain, yTrain, validation_data=(memoryTest, yTest), batch_size=32,
                                      epochs=2000,
                                      verbose=1,
                                      callbacks=callbacks)

                #plt.plot(history.history['acc'])
                #plt.plot(history.history['val_acc'])
                #plt.legend(['train', 'test'], loc='upper left')
                #plt.show()

                # Load best weights, predict, and save
                model_1.load_weights('model1_TESTWeights.txt')
                yPredTEST = model_1.predict(memoryTest)
                # spio.savemat('model_4_fold_1.mat', mdict={'yPred10': yPredTEST})

                # Save the binary results
                tempResults = N.zeros((yTest.shape[0], 2), dtype=N.float64)
                tempResults[:, 0] = yTest.reshape(yTest.shape[0])
                tempResults[:, 1] = yPredTEST.reshape(yTest.shape[0])

                if (rep == 1) & (fold == 1):
                    model_results = tempResults
                    spio.savemat('Model 1 Results', mdict={'predictions': model_results})
                else:
                    model_results = N.concatenate([model_results, tempResults])
                    spio.savemat('Model 1 Results', mdict={'predictions': model_results})
                    print(model_results)
                    print(tempResults)

            spio.savemat('Model 1 Results', mdict={'predictions': model_results})

            #plt.plot(history.history['acc'])
            #plt.plot(history.history['val_acc'])
            #plt.legend(['train', 'test'], loc='upper left')
            #plt.show()
            #print(model_results)
            #plt.plot(history.history['loss'])
            #plt.plot(history.history['val_loss'])
            #plt.legend(['train', 'test'], loc='upper left')
            #plt.show()

    elif modelNum == 2:
        modStr = str(modelNum)
        modelDataString = "MSL_Deep_Learning_Model_" + modStr + ".mat"  # Specify Model, Repetition, and Fold of Data File

        # Load the data
        modelData = h5py.File(modelDataString)  # Load Model Data File
        voiceMatrix = modelData["voiceMatrix"]
        Y = modelData["Y"]

        nSubs = Y.shape[1]
        print("There are " + str(nSubs) + " participants")
        nFolds = 10

        for rep in range(1,2):

            # Need to Balance the groups
            balancedIndices = um.balance_data(Y[0])
            voiceMatrix = voiceMatrix[:, balancedIndices]
            Y = Y[:, balancedIndices]
            nBalSubs = len(balancedIndices)

            idx = um.cross_val_inds(nBalSubs, nFolds)

            for fold in range (1,4):
                print("Rep: " + str(rep) + " Fold: " + str(fold))
                trainIdx, testIdx = um.kfold_train_test_idx(idx, fold)

                # Training Data
                voiceTrain = voiceMatrix[:, trainIdx]
                yTrain = Y[:, trainIdx]

                augData = um.data_augmentation_1D(voiceTrain)
                voiceTrain = augData

                # Testing Data
                voiceTest = voiceMatrix[:, testIdx]
                yTest = Y[:, testIdx]

                # Now Implement the model
                model_2 = blMods.model_2_cnn(voiceMatrix)

                # Reshape the TRAINING matrices
                # Voice data is 2D
                voiceX, voiceY = voiceTrain.shape
                B = N.reshape(voiceTrain, [voiceY * voiceX])
                voiceTrain = N.reshape(B, [voiceY, voiceX], [voiceY, voiceX])
                voiceTrain = N.expand_dims(voiceTrain, axis=2)

                # Reshape the TESTING matrices
                # Voice data is 2D
                voiceX, voiceY = voiceTest.shape
                B = N.reshape(voiceTest, [voiceY * voiceX])
                voiceTest = N.reshape(B, [voiceY, voiceX], [voiceY, voiceX])
                voiceTest = N.expand_dims(voiceTest, axis=2)

                # Reshape the TRAINING response vector
                z, y = N.shape(yTrain)
                yTrain = N.reshape(yTrain, [y, z], [y, z])

                # Reshape the TESTING response vector
                z, y = N.shape(yTest)
                yTest = N.reshape(yTest, [y, z], [y, z])

                # Now can fit the model
                # Set up the callbacks for saving the model with the best performance
                callbacks = [ModelCheckpoint('model_2_Weights.txt', monitor='val_acc', save_best_only=True, verbose=2)]

                history = model_2.fit(voiceTrain, yTrain, validation_data=(voiceTest, yTest), batch_size=32, epochs=2000, verbose=1,
                            callbacks=callbacks)

                # Load best weights, predict, and save
                model_2.load_weights('model_2_Weights.txt')
                yPredTEST = model_2.predict(voiceTest)

                # Save the binary results
                tempResults = N.zeros((yTest.shape[0], 2), dtype=N.float64)
                tempResults[:, 0] = yTest.reshape(yTest.shape[0])
                tempResults[:, 1] = yPredTEST.reshape(yTest.shape[0])

                if (rep == 1) & (fold == 1):
                    model_results = tempResults
                    spio.savemat('Model 2 Results', mdict={'predictions': model_results})
                else:
                    model_results = N.concatenate([model_results, tempResults])
                    spio.savemat('Model 2 Results', mdict={'predictions': model_results})
                    print(model_results)
                    print(tempResults)

            print(model_results)
            spio.savemat('Model 2 Results', mdict={'predictions': model_results})

            #plt.plot(history.history['acc'])
            #plt.plot(history.history['val_acc'])
            #plt.legend(['train', 'test'], loc='upper left')
            #plt.title('Model 2 Accuracy')
            #plt.show()
            #print(model_results)
            #plt.plot(history.history['loss'])
            #plt.plot(history.history['val_loss'])
            #plt.legend(['train', 'test'], loc='upper left')
            #plt.title('Model 2 Loss')
            #plt.show()

    elif modelNum == 3:
        modStr = str(modelNum)

    elif modelNum == 4:
        modStr = str(modelNum)
        modStr = str(modelNum)
        modelDataString = "MSL_Deep_Learning_Model_" + modStr + ".mat"  # Specify Model, Repetition, and Fold of Data File

        # Load the data
        modelData = h5py.File(modelDataString)  # Load Model Data File
        walkingMatrix = modelData["walkingMatrix"]
        Y = modelData["Y"]

        nSubs = Y.shape[1]
        print("There are " + str(nSubs) + " participants")
        nFolds = 10

        for rep in range(1, 2):

            # Need to Balance the groups
            balancedIndices = um.balance_data(Y[0])
            print(walkingMatrix.shape)
            walkingMatrix = walkingMatrix[:, :, balancedIndices]
            print(walkingMatrix.shape)
            Y = Y[:, balancedIndices]
            nBalSubs = len(balancedIndices)

            idx = um.cross_val_inds(nBalSubs, nFolds)

            for fold in range(1, 4):
                print("Rep: " + str(rep) + " Fold: " + str(fold))
                trainIdx, testIdx = um.kfold_train_test_idx(idx, fold)

                # Training Data
                walkingTrain = walkingMatrix[:, :, trainIdx]
                yTrain = Y[:, trainIdx]

                # Testing Data
                walkingTest = walkingMatrix[:, :, testIdx]
                yTest = Y[:, testIdx]

                # Implement the model
                model_4 = blMods.model_4_cnn(walkingMatrix)

                # Reshape the TRAINING matrices
                walkingZ, walkingX, walkingY = walkingTrain.shape
                B = N.reshape(walkingTrain, [walkingY * walkingX * walkingZ])
                walkingTrain = N.reshape(B, [walkingY, walkingX, walkingZ], [walkingY, walkingX, walkingZ])

                # Reshape the TESTING matrices
                walkingZ, walkingX, walkingY = walkingTest.shape
                B = N.reshape(walkingTest, [walkingY * walkingX * walkingZ])
                walkingTest = N.reshape(B, [walkingY, walkingX, walkingZ], [walkingY, walkingX, walkingZ])

                # Reshape the TRAINING response vector
                z, y = N.shape(yTrain)
                yTrain = N.reshape(yTrain, [y, z], [y, z])

                # Reshape the TESTING response vector
                z, y = N.shape(yTest)
                yTest = N.reshape(yTest, [y, z], [y, z])

                # Now can fit the model
                # Set up the callbacks for saving the model with the best performance
                callbacks = [ModelCheckpoint('model_4_TESTWeights.txt', monitor='val_acc', save_best_only=True, verbose=2)]

                history = model_4.fit(walkingTrain, yTrain, validation_data=(walkingTest, yTest), batch_size=32, epochs=2000,
                            verbose=1,
                            callbacks=callbacks)

                # Load best weights, predict, and save
                model_4.load_weights('model_4_TESTWeights.txt')
                yPredTEST = model_4.predict(walkingTest)
                #

                # Save the binary results
                tempResults = N.zeros((yTest.shape[0], 2), dtype=N.float64)
                tempResults[:, 0] = yTest.reshape(yTest.shape[0])
                tempResults[:, 1] = yPredTEST.reshape(yTest.shape[0])

                if (rep == 1) & (fold == 1):
                    model_results = tempResults
                    spio.savemat('Model 4 Results.mat', mdict={'predictions': model_results})
                else:
                    model_results = N.concatenate([model_results, tempResults])
                    spio.savemat('Model 4 Results.mat', mdict={'predictions': model_results})
                    print(model_results)
                    print(tempResults)

            #plt.plot(history.history['acc'])
            #plt.plot(history.history['val_acc'])
            #plt.legend(['train', 'test'], loc='upper left')
            #plt.show()
            #print(model_results)
            #plt.plot(history.history['loss'])
            #plt.plot(history.history['val_loss'])
            #plt.legend(['train', 'test'], loc='upper left')
            #plt.show()

    elif modelNum == 5:
        modStr = str(modelNum)

    elif modelNum == 6:
        modStr = str(modelNum)

    elif modelNum == 7:
        modStr = str(modelNum)

    elif modelNum == 8:
        modStr = str(modelNum)
        modelDataString = "MSL_Deep_Learning_Model_" + modStr + ".mat"  # Specify Model, Repetition, and Fold of Data File

        # Load the data
        modelData = h5py.File(modelDataString)  # Load Model Data File
        tappingMatrix = modelData["tappingMatrix"]
        Y = modelData["Y"]

        nSubs = Y.shape[1]
        print("There are " + str(nSubs) + " participants")
        nFolds = 10

        for rep in range(1, 2):

            # Need to Balance the groups
            balancedIndices = um.balance_data(Y[0])
            tappingMatrix = tappingMatrix[:, :, balancedIndices]
            Yt = Y[:, balancedIndices]
            nBalSubs = len(balancedIndices)

            idx = um.cross_val_inds(nBalSubs, nFolds)

            for fold in range(1, 4):
                print("Rep: " + str(rep) + " Fold: " + str(fold))
                trainIdx, testIdx = um.kfold_train_test_idx(idx, fold)

                # Training Data
                tappingTrain = tappingMatrix[:, :, trainIdx]
                yTrain = Yt[:, trainIdx]

                # Testing Data
                tappingTest = tappingMatrix[:, :, testIdx]
                yTest = Yt[:, testIdx]

                # Implement the model
                model_8 = blMods.model_8_cnn(tappingMatrix)

                # Reshape the TRAINING matrices
                tappingZ, tappingX, tappingY = tappingTrain.shape
                B = N.reshape(tappingTrain, [tappingY * tappingX * tappingZ])
                tappingTrain = N.reshape(B, [tappingY, tappingX, tappingZ], [tappingY, tappingX, tappingZ])

                # Reshape the TESTING matrices
                tappingZ, tappingX, tappingY = tappingTest.shape
                B = N.reshape(tappingTest, [tappingY * tappingX * tappingZ])
                tappingTest = N.reshape(B, [tappingY, tappingX, tappingZ], [tappingY, tappingX, tappingZ])

                # Reshape the TRAINING response vector
                z, y = N.shape(yTrain)
                yTrain = N.reshape(yTrain, [y, z], [y, z])

                # Reshape the TESTING response vector
                z, y = N.shape(yTest)
                yTest = N.reshape(yTest, [y, z], [y, z])

                # Now can fit the model
                # Set up the callbacks for saving the model with the best performance
                callbacks = [ModelCheckpoint('model8_TESTWeights.txt', monitor='val_acc', save_best_only=True, verbose=2)]

                history = model_8.fit(tappingTrain, yTrain, validation_data=(tappingTest, yTest), batch_size=32,
                                      epochs=2000,
                                      verbose=1,
                                      callbacks=callbacks)

                #plt.plot(history.history['acc'])
                #plt.plot(history.history['val_acc'])
                #plt.legend(['train', 'test'], loc='upper left')
                #plt.show()

                # Load best weights, predict, and save
                model_8.load_weights('model8_TESTWeights.txt')
                yPredTEST = model_8.predict(tappingTest)


                # Save the binary results
                tempResults = N.zeros((yTest.shape[0], 2), dtype=N.float64)
                tempResults[:, 0] = yTest.reshape(yTest.shape[0])
                tempResults[:, 1] = yPredTEST.reshape(yTest.shape[0])

                if (rep == 1) & (fold == 1):
                    model_results = tempResults
                    spio.savemat('Model 8 Results.mat', mdict={'predictions': model_results})
                else:
                    model_results = N.concatenate([model_results, tempResults])
                    spio.savemat('Model 8 Results.mat', mdict={'predictions': model_results})
                    print(model_results)
                    print(tempResults)

            #plt.plot(history.history['acc'])
            #plt.plot(history.history['val_acc'])
            #plt.legend(['train', 'test'], loc='upper left')
            #plt.show()
            #print(model_results)
            #plt.plot(history.history['loss'])
            #plt.plot(history.history['val_loss'])
            #plt.legend(['train', 'test'], loc='upper left')
            #plt.show()

    elif modelNum == 9:
        modStr = str(modelNum)

    elif modelNum == 10:
        modStr = str(modelNum)
        modelDataString = "MSL_Deep_Learning_Model_" + modStr + ".mat"  # Specify Model, Repetition, and Fold of Data File

        # Load the data
        modelData = h5py.File(modelDataString)  # Load Model Data File
        tappingMatrix = modelData["tappingMatrix"]
        voiceMatrix = modelData["voiceMatrix"]
        Y = modelData["Y"]

        nSubs = Y.shape[1]
        print("There are " + str(nSubs) + " participants")
        nFolds = 10

        for rep in range(1, 2):

            # Need to Balance the groups
            balancedIndices = um.balance_data(Y[0])
            tappingMatrix = tappingMatrix[:, :, balancedIndices]
            voiceMatrix = voiceMatrix[:, balancedIndices]
            Y = Y[:, balancedIndices]
            nBalSubs = len(balancedIndices)

            idx = um.cross_val_inds(nBalSubs, nFolds)

            for fold in range(1, 4):
                print("Rep: " + str(rep) + " Fold: " + str(fold))
                trainIdx, testIdx = um.kfold_train_test_idx(idx, fold)

                # Training Data
                tappingTrain = tappingMatrix[:, :, trainIdx]
                voiceTrain = voiceMatrix[:, trainIdx]
                yTrain = Y[:, trainIdx]

                # Testing Data
                tappingTest = tappingMatrix[:, :, testIdx]
                voiceTest = voiceMatrix[:, testIdx]
                yTest = Y[:, testIdx]

                # Implement the model
                model_10 = blMods.model_10_cnn(tappingMatrix, voiceMatrix)

                # Reshape the TRAINING matrices
                tappingZ, tappingX, tappingY = tappingTrain.shape
                B = N.reshape(tappingTrain, [tappingY * tappingX * tappingZ])
                tappingTrain = N.reshape(B, [tappingY, tappingX, tappingZ], [tappingY, tappingX, tappingZ])

                voiceX, voiceY = voiceTrain.shape
                B = N.reshape(voiceTrain, [voiceY * voiceX])
                voiceTrain = N.reshape(B, [voiceY, voiceX], [voiceY, voiceX])
                voiceTrain = N.expand_dims(voiceTrain, axis=2)

                # Reshape the TESTING matrices
                tappingZ, tappingX, tappingY = tappingTest.shape
                B = N.reshape(tappingTest, [tappingY * tappingX * tappingZ])
                tappingTest = N.reshape(B, [tappingY, tappingX, tappingZ], [tappingY, tappingX, tappingZ])

                voiceX, voiceY = voiceTest.shape
                B = N.reshape(voiceTest, [voiceY * voiceX])
                voiceTest = N.reshape(B, [voiceY, voiceX], [voiceY, voiceX])
                voiceTest = N.expand_dims(voiceTest, axis=2)

                # Reshape the TRAINING response vector
                z, y = N.shape(yTrain)
                yTrain = N.reshape(yTrain, [y, z], [y, z])

                # Reshape the TESTING response vector
                z, y = N.shape(yTest)
                yTest = N.reshape(yTest, [y, z], [y, z])

                # Now can fit the model
                # Set up the callbacks for saving the model with the best performance
                callbacks = [
                    ModelCheckpoint('model10_TESTWeights.txt', monitor='val_acc', save_best_only=True, verbose=2)]

                history = model_10.fit([tappingTrain, voiceTrain], yTrain, validation_data=([tappingTest, voiceTest], yTest),
                                      batch_size=32, epochs=1500, verbose=1, callbacks=callbacks)


                # Load best weights, predict, and save
                model_10.load_weights('model10_TESTWeights.txt')
                yPredTEST = model_10.predict([tappingTest, voiceTest])


                # Save the binary results
                tempResults = N.zeros((yTest.shape[0], 2), dtype=N.float64)
                tempResults[:, 0] = yTest.reshape(yTest.shape[0])
                tempResults[:, 1] = yPredTEST.reshape(yTest.shape[0])

                if (rep == 1) & (fold == 1):
                    model_results = tempResults
                    spio.savemat('Model 10 Results', mdict={'predictions': model_results})
                else:
                    model_results = N.concatenate([model_results, tempResults])
                    print(model_results)
                    print(tempResults)
                    spio.savemat('Model 10 Results', mdict={'predictions': model_results})

            #plt.plot(history.history['acc'])
            #plt.plot(history.history['val_acc'])
            #plt.legend(['train', 'test'], loc='upper left')
            #plt.show()
            #print(model_results)
            #plt.plot(history.history['loss'])
            #plt.plot(history.history['val_loss'])
            #plt.legend(['train', 'test'], loc='upper left')
            #plt.show()

    elif modelNum == 11:
        modStr = str(modelNum)

    elif modelNum == 12:
        modStr = str(modelNum)

    elif modelNum == 13:
        modStr = str(modelNum)

    elif modelNum == 14:
        modStr = str(modelNum)
        modelDataString = "MSL_Deep_Learning_Model_" + modStr + ".mat"  # Specify Model, Repetition, and Fold of Data File

        # Load the data
        modelData = h5py.File(modelDataString)  # Load Model Data File
        tappingMatrix = modelData["tappingMatrix"]
        walkingMatrix = modelData["walkingMatrix"]
        voiceMatrix = modelData["voiceMatrix"]
        Y = modelData["Y"]

        nSubs = Y.shape[1]
        print("There are " + str(nSubs) + " participants")
        nFolds = 10

        for rep in range(1, 2):

            # Need to Balance the groups
            balancedIndices = um.balance_data(Y[0])
            tappingMatrix = tappingMatrix[:, :, balancedIndices]
            walkingMatrix = walkingMatrix[:, :, balancedIndices]
            voiceMatrix = voiceMatrix[:, balancedIndices]
            Y = Y[:, balancedIndices]
            nBalSubs = len(balancedIndices)

            idx = um.cross_val_inds(nBalSubs, nFolds)

            for fold in range(1, 4):
                print("Rep: " + str(rep) + " Fold: " + str(fold))
                trainIdx, testIdx = um.kfold_train_test_idx(idx, fold)

                # Training Data
                tappingTrain = tappingMatrix[:, :, trainIdx]
                walkingTrain = walkingMatrix[:, :, trainIdx]
                voiceTrain = voiceMatrix[:, trainIdx]
                yTrain = Y[:, trainIdx]

                augData = um.data_augmentation_1D(voiceTrain)

                for i in range(0,16):
                    f, axarr = plt.subplots(2, sharex=True)
                    f.suptitle('Augmented Voice Data')
                    axarr[0].plot(voiceTrain[:, i])
                    axarr[1].plot(augData[:, i])
                    plt.show()

                # Testing Data
                tappingTest = tappingMatrix[:, :, testIdx]
                walkingTest = walkingMatrix[:, :, testIdx]
                voiceTest = voiceMatrix[:, testIdx]
                yTest = Y[:, testIdx]

                # Now Implement the model
                model_14 = blMods.model_14_cnn(tappingMatrix, walkingMatrix, voiceMatrix)

                # Reshape the TRAINING matrices
                tappingZ, tappingX, tappingY = tappingTrain.shape
                B = N.reshape(tappingTrain, [tappingY * tappingX * tappingZ])
                tappingTrain = N.reshape(B, [tappingY, tappingX, tappingZ], [tappingY, tappingX, tappingZ])

                voiceX, voiceY = voiceTrain.shape
                B = N.reshape(voiceTrain, [voiceY * voiceX])
                voiceTrain = N.reshape(B, [voiceY, voiceX], [voiceY, voiceX])
                voiceTrain = N.expand_dims(voiceTrain, axis=2)

                walkingZ, walkingX, walkingY = walkingTrain.shape
                B = N.reshape(walkingTrain, [walkingY * walkingX * walkingZ])
                walkingTrain = N.reshape(B, [walkingY, walkingX, walkingZ], [walkingY, walkingX, walkingZ])

                # Reshape the TESTING matrices
                tappingZ, tappingX, tappingY = tappingTest.shape
                B = N.reshape(tappingTest, [tappingY * tappingX * tappingZ])
                tappingTest = N.reshape(B, [tappingY, tappingX, tappingZ], [tappingY, tappingX, tappingZ])

                voiceX, voiceY = voiceTest.shape
                B = N.reshape(voiceTest, [voiceY * voiceX])
                voiceTest = N.reshape(B, [voiceY, voiceX], [voiceY, voiceX])
                voiceTest = N.expand_dims(voiceTest, axis=2)

                walkingZ, walkingX, walkingY = walkingTest.shape
                B = N.reshape(walkingTest, [walkingY * walkingX * walkingZ])
                walkingTest = N.reshape(B, [walkingY, walkingX, walkingZ], [walkingY, walkingX, walkingZ])

                # Reshape the TRAINING response vector
                z, y = N.shape(yTrain)
                yTrain = N.reshape(yTrain, [y, z], [y, z])

                # Reshape the TESTING response vector
                z, y = N.shape(yTest)
                yTest = N.reshape(yTest, [y, z], [y, z])

                # Now can fit the model
                # Set up the callbacks for saving the model with the best performance
                callbacks = [ModelCheckpoint('model_14_Weights.txt', monitor='val_acc', save_best_only=True, verbose=2)]

                history = model_14.fit([tappingTrain, walkingTrain, voiceTrain], yTrain, validation_data=([tappingTest, walkingTest, voiceTest], yTest), batch_size=32,
                                      epochs=2000, verbose=1,
                                      callbacks=callbacks)

                # Load best weights, predict, and save
                model_14.load_weights('model_14_Weights.txt')
                yPredTEST = model_14.predict(voiceTest)

                # Save the binary results
                tempResults = N.zeros((yTest.shape[0], 2), dtype=N.float64)
                tempResults[:, 0] = yTest.reshape(yTest.shape[0])
                tempResults[:, 1] = yPredTEST.reshape(yTest.shape[0])

                if (rep == 1) & (fold == 1):
                    model_results = tempResults
                    spio.savemat('Model 14 Results', mdict={'predictions': model_results})
                else:
                    model_results = N.concatenate([model_results, tempResults])
                    spio.savemat('Model 14 Results', mdict={'predictions': model_results})
                    print(model_results)
                    print(tempResults)

            print(model_results)
            spio.savemat('Model 14 Results', mdict={'predictions': model_results})

    elif modelNum == 15:
        """ Model for target domain [1 1 1 1]"""

        #modStr = str(modelNum)
        #repStr = str(kRep)
        #foldStr = str(fold)
        #modelDataString = "rawModelData_Mod_" + modStr + "_Rep_" + repStr + "_Fold_" + foldStr + ".mat"
        modelDataString = "rawModelData_15.mat"  # Specify Model, Repetition, and Fold of Data File
        print(modelDataString)

        # Load the data
        modelData = h5py.File(modelDataString)  # Load Model Data File
        tappingMatrix = modelData["Tapping"]
        walkingMatrix = modelData["Gait"]
        voiceMatrix = modelData["Voice"]
        memoryMatrix = modelData["Memory"]
        Y = modelData["Y"]

        # Load the model
        model_15 = blMods.model_15_cnn(tappingMatrix, walkingMatrix, voiceMatrix, memoryMatrix)

        # Reshape the original data matrices
        # For 3D Tapping Data
        tapZ, tapX, tapY = tappingMatrix.shape
        B = N.reshape(tappingMatrix, [tapY * tapX * tapZ])
        tappingMatrix = N.reshape(B, [tapY, tapX, tapZ], [tapY, tapX, tapZ])

        # For 3D Gait Data
        gaitZ, gaitX, gaitY = walkingMatrix.shape
        B = N.reshape(walkingMatrix, [gaitY * gaitX * gaitZ])
        walkingMatrix = N.reshape(B, [gaitY, gaitX, gaitZ], [gaitY, gaitX, gaitZ])

        # Voice data is only 2D
        voiceX, voiceY = voiceMatrix.shape
        B = N.reshape(voiceMatrix, [voiceY * voiceX])
        voiceMatrix = N.reshape(B, [voiceY, voiceX], [voiceY, voiceX])
        voiceMatrix = N.expand_dims(voiceMatrix, axis=2)

        # memory data is only 2D
        memoryX, memoryY = memoryMatrix.shape
        B = N.reshape(memoryMatrix, [memoryY * memoryX])
        memoryMatrix = N.reshape(B, [memoryY, memoryX], [memoryY, memoryX])
        memoryMatrix = N.expand_dims(memoryMatrix, axis=2)

        # Reshape the response vector
        z, y = N.shape(Y)
        Y = N.reshape(Y, [y, z], [y, z])

        print(tappingMatrix.shape)
        print(walkingMatrix.shape)
        print(voiceMatrix.shape)
        print(memoryMatrix.shape)

        # Now can fit the model
        model_15.fit([tappingMatrix, walkingMatrix, voiceMatrix, memoryMatrix], Y, batch_size=30, epochs=10000,
                     verbose=1, shuffle=True)


    # save the results of the model
    resultsStr = "model_" + str(modelNum) + "_kfold_results.mat"
    spio.savemat(resultsStr, mdict={'predictions': model_results})


