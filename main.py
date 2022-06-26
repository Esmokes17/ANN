import numpy as np
from PIL import Image

from ANN import *
from loadData import loadCsv
from model import Model

def main():
    X, Y = loadCsv("datasets/mnist_train.csv")

    # normalize and select data
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize]
    Y = Y[randomize]
    X = X[:2000, :]
    X = X / 255.
    Y = Y[:2000,:]

    n_samples = X.shape[0]
    n_features = X.shape[1]
    print("number of test samples ", n_samples)
    print("number of features     ", n_features)

    networks = [
        Dense(n_features, 128),
        Activation(type = typeActivation.RELU),
        Dense(128, 10),
        Activation(type = typeActivation.SOFTMAX)
    ]

    model = Model(networks, mse, d_mse)
    model.fit(X, Y)

    model.save("test.pkl")
    model = Model.load("test.pkl")

    X_test, Y_test = loadCsv("datasets/mnist_train.csv")
    accuracy = model.evaluate(X_test, Y_test)
    print(f"accuracy of test {accuracy*100}")

if __name__ == '__main__':
    main()
