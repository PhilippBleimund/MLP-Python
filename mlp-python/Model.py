from Layer import InputLayer, PerceptronLayer, PredictionLayer, _Layer


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

    def __call__(self, input_data):
        self.input_layer.set_data(input_data)
        self.output_layer.evaluate_layer()

        return self.output_layer._get_output()
