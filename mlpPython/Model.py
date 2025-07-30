from .Layer import InputLayer, PerceptronLayer, PredictionLayer, _Layer
import numpy as np


class Model:
    def __init__(self):
        self.input_layer: InputLayer
        self.hidden_layer: list[PerceptronLayer]
        self.output_layer: PredictionLayer

        self.full_layer_model = list[_Layer]

        self.to_assemble = []

    def add(self, layerdata: _Layer):
        self.to_assemble.append(layerdata)

    def assemble_model(self):
        # get input layer
        input_layers = [layer for layer in self.to_assemble if isinstance(layer, InputLayer)]
        if len(input_layers) != 1:
            raise ValueError("There must be exactly one input layer.")
        self.input_layer = input_layers[0]

        # get hidden layers
        hidden_layers = [layer for layer in self.to_assemble if isinstance(layer, PerceptronLayer)]
        self.hidden_layers = hidden_layers

        # get output layer
        output_layers = [layer for layer in self.to_assemble if isinstance(layer, PredictionLayer)]
        if len(output_layers) != 1:
            raise ValueError("There must be exactly one output layer.")
        self.output_layer = output_layers[0]

        self.full_layer_model = [self.input_layer] + self.hidden_layers + [self.output_layer]

        # assemble the layers
        for i, layer in enumerate(self.full_layer_model):
            prev_layer = self.full_layer_model[i - 1] if i > 0 else None
            next_layer = self.full_layer_model[i +
                                               1] if i < len(self.full_layer_model) - 1 else None
            layer.link_layer(prev_layer, next_layer)

    def set_training_settings(self, batch_size):
        self.max_batch_size = batch_size

    def train_model(self, X, Y, epochs: int, X_test=None, Y_test=None):
        if not self.max_batch_size:
            raise ValueError("missing trainings settings. Use Model.set_training_settings()")

        # prepare layers for training
        for layer in self.full_layer_model:
            layer.prepare_for_training(self.max_batch_size)

        X, Y = np.copy(X), np.copy(Y)

        rng = np.random.default_rng(seed=1)
        for i in range(epochs):
            p = rng.permutation(len(X))

            for i in range(int(len(X)/self.max_batch_size)):
                print(i)
                self.input_layer.set_data(
                    X[p[i*self.max_batch_size:(i+1)*self.max_batch_size]], self.max_batch_size)
                self.output_layer.evaluate_layer(self.max_batch_size)
                self.output_layer.train_layer(
                    self.max_batch_size, correct_solution_idx=Y[p[i*self.max_batch_size:(i+1)*self.max_batch_size]])
            if X_test is not None and Y_test is not None:
                # test accuracy on test data
                pred = self.__call__(X_test, output_as_idx=True)
                print(f"accuracy: {np.sum(pred == Y_test)/len(Y_test)}")

        # from .Layer import lp
        # lp.print_stats()

    def __call__(self, input_data, output_as_idx=None):
        if input_data.ndim == 1:
            self.input_layer.set_data(input_data, 1)
            self.output_layer.evaluate_layer(1)
            if output_as_idx:
                return self.output_layer._get_output()[0]
            else:
                return self.output_layer.get_prediction()[0]
        else:
            output = []
            input_size = len(input_data)
            processing_start = 0

            while processing_start < input_size:
                processing_step = min(self.max_batch_size, input_size - processing_start)
                self.input_layer.set_data(
                    input_data[processing_start:processing_start + processing_step, :],
                    processing_step
                )
                self.output_layer.evaluate_layer(processing_step)

                if output_as_idx:
                    output += self.output_layer._get_output()[:processing_step]
                else:
                    output += self.output_layer.get_prediction()[:processing_step]

                processing_start += processing_step

            return output
