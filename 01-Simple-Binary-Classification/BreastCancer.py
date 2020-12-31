
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense


rawData = pd.read_csv("wdbc.data")
rawData.info()

predictors = rawData.loc[:, 'radius_mean':'fractal_dimension_worst'].values
target  = rawData.loc[:, 'diagnosis'].values


labelencoder = LabelEncoder()
target = labelencoder.fit_transform(target)


predictorsTraining, predictorsTest, targetTraining, targetTest = train_test_split(predictors, target, test_size=0.25)

neuralNetwork = Sequential()
neuralNetwork.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30))
neuralNetwork.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'))
neuralNetwork.add(Dense(units = 1, activation = 'sigmoid'))

optimizer = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)

neuralNetwork.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

neuralNetwork.fit(predictorsTraining, targetTraining, batch_size = 10, epochs = 100)

forecasts = neuralNetwork.predict(predictorsTest)
forecasts = (forecasts > 0.5)
from sklearn.metrics import confusion_matrix, accuracy_score
precision = accuracy_score(targetTest, forecasts)
matriz = confusion_matrix(targetTest, forecasts)
print(matriz)