import pandas as pd
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


rawData = pd.read_csv("wdbc.data")
rawData.info()

predictors = rawData.loc[:, 'radius_mean':'fractal_dimension_worst'].values
target  = rawData.loc[:, 'diagnosis'].values


labelencoder = LabelEncoder()
target = labelencoder.fit_transform(target)

def creatingNeuralNetwork(optimizer, loos, kernel_initializer, activation, neurons):
    neuralNetwork = Sequential()
    neuralNetwork.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer, input_dim = 30))
    neuralNetwork.add(Dropout(0.3))
    neuralNetwork.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer))
    neuralNetwork.add(Dropout(0.1))
    neuralNetwork.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer))
    neuralNetwork.add(Dropout(0.3))
    neuralNetwork.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer))
    neuralNetwork.add(Dropout(0.1))
    neuralNetwork.add(Dense(units = 1, activation = 'sigmoid'))
    neuralNetwork.compile(optimizer = optimizer, loss = loos, metrics = ['binary_accuracy'])
    return neuralNetwork

neuralNetwork = KerasClassifier(build_fn = creatingNeuralNetwork)

parameters = {'batch_size': [10, 30],
              'epochs': [50, 100],
              'optimizer': ['adam', 'sgd'],
              'loos': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [16, 8]}
grid_search = GridSearchCV(estimator = neuralNetwork,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5)

grid_search = grid_search.fit(predictors, target)
best_parameters = grid_search.best_params_
best_precision = grid_search.best_score_ 