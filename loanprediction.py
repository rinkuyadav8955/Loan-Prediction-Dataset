

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train_set.csv')
X = dataset.iloc[:, 1:12].values
y = dataset.iloc[:, [12]].values

dataset_test= pd.read_csv('test_set.csv')
X_test = dataset_test.iloc[:, 1:12].values

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
X[:,[2,5,6,7,8,9]]=imputer.fit_transform(X[:,[2,5,6,7,8,9]])
X_test[:,[2,5,6,7,8,9]]=imputer.fit_transform(X_test[:,[2,5,6,7,8,9]])


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_0 = LabelEncoder()
X[:, 0] = labelencoder_X_0.fit_transform(X[:, 0])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])
labelencoder_X_10 = LabelEncoder()
X[:,10] = labelencoder_X_10.fit_transform(X[:, 10])

onehotencoder = OneHotEncoder(categorical_features = [10])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]



# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_11 = LabelEncoder()
X_test[:, 1] = labelencoder_X_11.fit_transform(X_test[:, 1])
labelencoder_X_01 = LabelEncoder()
X_test[:, 0] = labelencoder_X_01.fit_transform(X_test[:, 0])
labelencoder_X_31 = LabelEncoder()
X_test[:, 3] = labelencoder_X_31.fit_transform(X_test[:, 3])
labelencoder_X_41 = LabelEncoder()
X_test[:, 4] = labelencoder_X_41.fit_transform(X_test[:, 4])
labelencoder_X_101 = LabelEncoder()
X_test[:,10] = labelencoder_X_101.fit_transform(X_test[:, 10])

onehotencoder = OneHotEncoder(categorical_features = [10])
X_test = onehotencoder.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X= sc.fit_transform(X)
X_test = sc.transform(X_test)



# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X, y)


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu', input_dim = 12))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X, y, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

y_train_pred=classifier.predict(X)
y_train_pred=(y_train_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_train_pred)























