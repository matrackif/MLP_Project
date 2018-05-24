from numpy import exp, array, random, dot, extract
from enum import Enum
from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import CategoricalEncoder
import utils
import wine_data, mushrooms_data, flags_data
import numpy
import pandas


class DataSource(Enum):
    WINE = 1
    MUSHROOMS = 2
    FLAGS = 3


class NeuronLayer:
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork:
    def __init__(self, data_source: DataSource = DataSource.WINE):
        self.input_layer = None
        self.hidden_layer1 = None
        self.hidden_layer2 = None
        self.output_layer = None
        self.data_source = data_source
        self.data = None
        if self.data_source == DataSource.WINE:
            self.data = wine_data.WineData()
            self.input_layer = NeuronLayer(20, 13)
            self.hidden_layer1 = NeuronLayer(3, 20)
            self.train(10000)
            input_layer_output, hidden_layer1_output = self.forward_prop(self.data.train_x)
            hidden_layer1_output = utils.classify(hidden_layer1_output)
            num_matches = 0
            for i in range(self.data.train_count):
                if (hidden_layer1_output[i] == self.data.train_y[i]).all():
                    num_matches += 1
            print('Training set performance:', float(num_matches) / float(self.data.train_count))
            num_matches = 0
            input_layer_output, hidden_layer1_output = self.forward_prop(self.data.test_x)
            hidden_layer1_output = utils.classify(hidden_layer1_output)
            print('hidden_layer1_output', hidden_layer1_output)
            print('self.test_y', self.data.test_y)
            for i in range(self.data.test_count):
                if (hidden_layer1_output[i].all() == self.data.test_y[i].all()).all():
                    num_matches += 1
            print('Test set performance:', float(num_matches) / float(self.data.test_count))
            self.normalize_and_classify(array([[14.13, 4.1, 2.74, 24.5, 96, 2.05, .76, .56, 1.35, 9.2, .61, 1.6, 560]]))
        elif self.data_source == DataSource.MUSHROOMS:
            pass
        elif self.data_source == DataSource.FLAGS:
            pass

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def sigmoid(self, x):
        return 1 / (1 + exp(-x))
        # return .5 * (1 + numpy.tanh(.5 * x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.forward_prop(self.data.train_x)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = self.data.train_y - output_from_layer_2
            layer2_delta = layer2_error * self.sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.hidden_layer1.synaptic_weights.T)
            layer1_delta = layer1_error * self.sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = self.data.train_x.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.input_layer.synaptic_weights += layer1_adjustment
            self.hidden_layer1.synaptic_weights += layer2_adjustment

    def forward_prop(self, inputs):
        output_from_layer1 = self.sigmoid(dot(inputs, self.input_layer.synaptic_weights))
        output_from_layer2 = self.sigmoid(dot(output_from_layer1, self.hidden_layer1.synaptic_weights))
        return output_from_layer1, output_from_layer2

    def normalize_and_classify(self, input_x):
        # call only after training
        for i in range(input_x.shape[1]):
            input_x[:, i] = (input_x[:, i] - self.data.means[i]) / self.data.standard_deviations[i]

        input_layer_output, hidden_layer1_output = self.forward_prop(input_x)
        hidden_layer1_output = utils.classify(hidden_layer1_output)
        print('normalize_and_classify() result:\n', hidden_layer1_output)

    # The neural network prints its weights
    def print_weights(self):
        print("Input layer weights: ")
        print(self.input_layer.synaptic_weights)
        print("Hidden layer 1 weights")
        print(self.hidden_layer1.synaptic_weights)