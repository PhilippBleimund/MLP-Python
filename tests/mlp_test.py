from mlpPython.Model import Model
from mlpPython.Layer import InputLayer, PerceptronLayer, PredictionLayer

from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np


def test_core():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    cifar_classes = ['airplane', 'automobile', 'bird', 'cat',
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Transform label indices to one-hot encoded vectors

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Transform images from (32,32,3) to 3072-dimensional vectors (32*32*3)

    X_train = np.reshape(X_train, (50000, 3072))
    X_test = np.reshape(X_test, (10000, 3072))
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalization of pixel values (to [0-1] range)

    X_train /= 255
    X_test /= 255


    model = Model()
    model.add(InputLayer(3072))
    model.add(PerceptronLayer(256, "sigmoid"))
    model.add(PerceptronLayer(256, "sigmoid"))
    model.add(PredictionLayer(10, "softmax", cifar_classes))
    model.assemble_model()
    model.train_model(X_train, y_train, 50, 0.02)

    print(model(X_test[0]))
