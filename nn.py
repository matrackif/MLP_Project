from numpy import exp, array, random, dot, extract
from enum import Enum
from sklearn.preprocessing import OneHotEncoder
import utils
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
        self.file = ''
        self.data_source = data_source
        if self.data_source == DataSource.WINE:
            self.file = 'wine.txt'
        elif self.data_source == DataSource.MUSHROOMS:
            self.file = 'mushrooms.txt'
        elif self.data_source == DataSource.FLAGS:
            self.file = 'flags.txt'
        self.data = pandas.read_csv(self.file)
        self.values = self.data.values
        self.nrows = self.values.shape[0]
        self.ncols = self.values.shape[1]
        print('Data:\n', self.data.head(), 'Num Rows:', self.nrows, 'Num Cols:', self.ncols)
        self.train_percentage = 0.8
        self.train_count = int(self.train_percentage * self.nrows)
        if self.data_source == DataSource.WINE:
            utils.normalize(self.values[:, 1:])
            print('self.values:\n', self.values)
            self.input_layer = NeuronLayer(20, 13)
            self.hidden_layer1 = NeuronLayer(3, 20)
            self.reg1 = self.values[self.values[:, 0] == 1]
            self.reg2 = self.values[self.values[:, 0] == 2]
            self.reg3 = self.values[self.values[:, 0] == 3]

            reg1_train_count = int(self.train_percentage * self.reg1.shape[0])
            reg2_train_count = int(self.train_percentage * self.reg2.shape[0])
            reg3_train_count = int(self.train_percentage * self.reg3.shape[0])

            self.train_x_reg1 = self.reg1[:reg1_train_count, 1:]
            self.train_x_reg2 = self.reg2[:reg2_train_count, 1:]
            self.train_x_reg3 = self.reg3[:reg3_train_count, 1:]

            self.train_y_reg1 = self.reg1[:reg1_train_count, 0].reshape(-1, 1)
            self.train_y_reg2 = self.reg2[:reg2_train_count, 0].reshape(-1, 1)
            self.train_y_reg3 = self.reg3[:reg3_train_count, 0].reshape(-1, 1)

            self.test_x_reg1 = self.reg1[reg1_train_count:, 1:]
            self.test_x_reg2 = self.reg2[reg2_train_count:, 1:]
            self.test_x_reg3 = self.reg3[reg3_train_count:, 1:]

            self.test_y_reg1 = self.reg1[reg1_train_count:, 0].reshape(-1, 1)
            self.test_y_reg2 = self.reg2[reg2_train_count:, 0].reshape(-1, 1)
            self.test_y_reg3 = self.reg3[reg3_train_count:, 0].reshape(-1, 1)

            self.train_x = numpy.empty((0,  self.train_x_reg1.shape[1]))
            self.train_x = numpy.append(self.train_x, self.train_x_reg1, axis=0)
            self.train_x = numpy.append(self.train_x, self.train_x_reg2, axis=0)
            self.train_x = numpy.append(self.train_x, self.train_x_reg3, axis=0)

            self.train_y = numpy.empty((0, 1))
            self.train_y = numpy.append(self.train_y, self.train_y_reg1, axis=0)
            self.train_y = numpy.append(self.train_y, self.train_y_reg2, axis=0)
            self.train_y = numpy.append(self.train_y, self.train_y_reg3, axis=0)

            self.test_x = numpy.empty((0,  self.test_x_reg1.shape[1]))
            self.test_x = numpy.append(self.test_x, self.test_x_reg1, axis=0)
            self.test_x = numpy.append(self.test_x, self.test_x_reg2, axis=0)
            self.test_x = numpy.append(self.test_x, self.test_x_reg3, axis=0)

            self.test_y = numpy.empty((0, 1))
            self.test_y = numpy.append(self.test_y, self.test_y_reg1, axis=0)
            self.test_y = numpy.append(self.test_y, self.test_y_reg2, axis=0)
            self.test_y = numpy.append(self.test_y, self.test_y_reg3, axis=0)

            print('train_x rows:', self.train_x.shape[0], 'train_x cols:', self.train_x.shape[1])
            print('train_y rows:', self.train_y.shape[0], 'train_y cols:', self.train_y.shape[1])
            print('test_x rows:', self.test_x.shape[0], 'test_x cols:', self.test_x.shape[1])
            print('test_y rows:', self.test_y.shape[0], 'test_y cols:', self.test_y.shape[1])

            print('\ntrain_x:\n', self.train_x)
            print('\ntrain_y:\n', self.train_y)
            print('\ntest_x:\n', self.test_x)
            print('\ntest_y:\n', self.test_y)
            # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
            enc = OneHotEncoder()
            enc.fit(self.train_y)
            # print('HOT:', enc.n_values_)
            # print('HOT2:', enc.transform([[4]]).toarray())
            self.train_y = enc.transform(self.train_y).toarray()
            self.test_y = enc.transform(self.test_y).toarray()
            # print('\ntrain_y:\n', self.train_y)
            self.train(10000)
            input_layer_output, hidden_layer1_output = self.forward_prop(self.train_x)
            hidden_layer1_output = utils.classify(hidden_layer1_output)
            print('hidden_layer1_output:', hidden_layer1_output)



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
            output_from_layer_1, output_from_layer_2 = self.forward_prop(self.train_x)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = self.train_y - output_from_layer_2
            layer2_delta = layer2_error * self.sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.hidden_layer1.synaptic_weights.T)
            layer1_delta = layer1_error * self.sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = self.train_x.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.input_layer.synaptic_weights += layer1_adjustment
            self.hidden_layer1.synaptic_weights += layer2_adjustment

    def forward_prop(self, inputs):
        output_from_layer1 = self.sigmoid(dot(inputs, self.input_layer.synaptic_weights))
        output_from_layer2 = self.sigmoid(dot(output_from_layer1, self.hidden_layer1.synaptic_weights))
        return output_from_layer1, output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        print("Input layer weights: ")
        print(self.input_layer.synaptic_weights)
        print("Hidden layer 1 weights")
        print(self.hidden_layer1.synaptic_weights)