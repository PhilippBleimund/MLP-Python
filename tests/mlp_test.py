from mlpPython import *

import numpy as np
import time

from .load_cifar_10 import load_cifar_10_data

import matplotlib.pyplot as plt


def test_core():
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        load_cifar_10_data()

    cifar_classes = ['airplane', 'automobile', 'bird', 'cat',
                     'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Transform images from (32,32,3) to 3072-dimensional vectors (32*32*3)

    X_train_flat = np.reshape(train_data, (50000, 3072))
    X_test_flat = np.reshape(test_data, (10000, 3072))
    X_train_flat = X_train_flat.astype('float32')
    X_test_flat = X_test_flat.astype('float32')

    # Normalization of pixel values (to [0-1] range)

    X_train_flat /= 255
    X_train = train_data / 255
    X_test_flat /= 255
    X_test = test_data / 255

    model = Model()
    model.add(InputLayer((32, 32, 3)))
    model.add(ConvolutionLayer((32, 32, 8)))
    model.add(Pool((8, 8)))
    model.add(FlatteningLayer())
    model.add(LinearLayer(4 * 4 * 8))
    model.add(DropoutLayer(4 * 4 * 8, 0.5))
    model.add(NormalizationLayer(4 * 4 * 8, "batch"))
    model.add(ActivationLayer(4 * 4 * 8, "relu"))
    model.add(LinearLayer(256))
    model.add(DropoutLayer(256, 0.5))
    model.add(NormalizationLayer(256, "batch"))
    model.add(ActivationLayer(256, "relu"))
    model.add(LinearLayer(10))
    model.add(PredictionLayer(10, cifar_classes))
    model.assemble_model()
    model.set_training_settings(batch_size=10, optimizer="sgd")
    metrics = model.train_model(X_train[:100], train_labels[:100],
                                2, X_test[:100], test_labels[:100])

    epochs = list(range(1, len(metrics["accuracy_train"]) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, metrics["accuracy_train"], label="Train Accuracy", marker='o')
    plt.plot(epochs, metrics["accuracy_test"], label="Test Accuracy", marker='s')
    plt.ylim(0, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Testing Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("now testing")
    model.lock_model(64)

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
