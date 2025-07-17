from Layer import InputLayer, _InputLayer, PerceptronLayer, _PerceptronLayer, PredictionLayer, _PredictionLayer


class Model:
    def __init__(self):
        self.input_layer = None
        self.hidden_layer = []
        self.output_layer = None

    def add_input_layer(self, input_size):
        self.input_layer =
