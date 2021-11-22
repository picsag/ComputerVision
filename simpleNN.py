import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras import backend as K
import tensorflow as tf

model = Sequential()

"""
For a problem of binary classification, the “pima-indians-diabetes” dataset will be used.
Download the dataset from:  https://data.world/uci/pima-indians-diabetes
Copy the .csv file containing the database in the project folder. 
The .csv file has 9 columns, the last one being 0 or 1, corresponding to weather the patient has or not diabetes. 
"""


def load_dataset(path: str):
    dataset = np.loadtxt(path, delimiter=",", skiprows=1)
    X_train = dataset[0: 500, 0: 8]
    Y_train = dataset[0: 500, 8]
    X_test = dataset[500: 768, 0: 8]
    Y_test = dataset[500: 768, 8]

    return X_train, Y_train, X_test, Y_test


def create_model(optimizer: str):
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(10, input_dim=12, activation='relu'))
    model.add(Dense(8, input_dim=10, activation='relu'))
    model.add(Dense(4, input_dim=8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


if __name__ == "__main__":
    model1 = create_model(optimizer='adam')
    model2 = create_model(optimizer='adagrad')

    print(model1.summary())

    # read the layer names from a model
    print("Layer names . . .")
    for layer in model2.layers:
        print(layer.name)

    X_train, Y_train, X_test, Y_test = load_dataset("data/pima-indians-diabetes.data.csv")

    print(f"Training model 1 . . .")
    model1.fit(X_train, Y_train, epochs=15, batch_size=3)

    print(f"Training model 2 . . .")
    model2.fit(X_train, Y_train, epochs=15, batch_size=3)

    scores1 = model1.evaluate(X_test, Y_test)
    print("\n%s: %.2f%%" % (model1.metrics_names[1], scores1[1] * 100))

    scores2 = model2.evaluate(X_test, Y_test)
    print("\n%s: %.2f%%" % (model2.metrics_names[1], scores2[1] * 100))