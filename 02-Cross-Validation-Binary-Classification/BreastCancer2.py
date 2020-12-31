import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier


rawData = pd.read_csv("wdbc.data")
rawData.info()

predictors = rawData.loc[:, 'radius_mean':'fractal_dimension_worst'].values
target  = rawData.loc[:, 'diagnosis'].values


labelencoder = LabelEncoder()
target = labelencoder.fit_transform(target)

def creatingNeuralNetwork():
    neuralNetwork = Sequential()
    neuralNetwork.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30))
    neuralNetwork.add(Dropout(0.2))
    neuralNetwork.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'))
    neuralNetwork.add(Dropout(0.2))
    neuralNetwork.add(Dense(units = 1, activation = 'sigmoid'))
    optimizer = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
    neuralNetwork.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    return neuralNetwork

neuralNetwork = KerasClassifier(build_fn = creatingNeuralNetwork, epochs = 100, batch_size = 10)
result = cross_val_score(estimator = neuralNetwork, X = predictors, y = target, cv = 10, scoring = 'accuracy')

#mean to calculate the mean of all Cross-Validation
mean = result.mean()
#if your standardDeviation return a high value, probabily your network is an overfitting network.
standardDeviation = result.std()