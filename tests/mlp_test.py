from mlpPython import Model, InputLayer, PerceptronLayer, PredictionLayer

import numpy as np
import time

from .load_cifar_10 import load_cifar_10_data


def test_core():
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        load_cifar_10_data()

    cifar_classes = ['airplane', 'automobile', 'bird', 'cat',
                     'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Transform images from (32,32,3) to 3072-dimensional vectors (32*32*3)

    X_train = np.reshape(train_data, (50000, 3072))
    X_test = np.reshape(test_data, (10000, 3072))
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalization of pixel values (to [0-1] range)

    X_train /= 255
    X_test /= 255

    model = Model()
    model.add(InputLayer(3072))
    model.add(PerceptronLayer(256, "relu"))
    model.add(PerceptronLayer(256, "relu"))
    model.add(PredictionLayer(10, cifar_classes))
    model.assemble_model()
    model.set_training_settings(batch_size=32)
    model.train_model(X_train, train_labels, 15, X_test, test_labels)

    print("now testing")

    # check acuracy
    time1 = time.time()
    num_correct = 0
    for i in range(len(X_test)):
        num_correct += model(X_test[i]) == cifar_classes[test_labels[i]]
    print(f"accuracy: {(num_correct/len(X_test))*100}%")
    print(f"time: {time.time()-time1}")

    time1 = time.time()
    model(X_test)
    print(f"time: {time.time()-time1}")

    print(model(X_test[0]))
