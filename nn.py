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
        self.weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork:
    def __init__(self, data_source: DataSource = DataSource.WINE):
        self.input_layer = None
        self.hidden_layer1 = None
        self.hidden_layer2 = None
        self.output_layer = None
        self.train_output = None
        self.test_output = None
        self.data_source = data_source
        self.data = None
        self.data_name = ''
        self.layers = []

        if self.data_source == DataSource.WINE:
            self.data = wine_data.WineData()
            self.train_wine_data()

        elif self.data_source == DataSource.MUSHROOMS:
            self.data = mushrooms_data.MushroomData()
            self.train_mushroom_data()

        elif self.data_source == DataSource.FLAGS:
            self.data = flags_data.FlagData()
            self.train_flag_data()

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))
        # return .5 * (1 + numpy.tanh(.5 * x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def train(self, num_iters):
        for iteration in range(num_iters):
            self.back_prop()

    def back_prop(self):
        outputs = [self.data.train_x]
        ret = self.forward_prop(self.data.train_x)
        for output in ret:
            outputs.append(output)

        layer_error = self.data.train_y - outputs[-1]
        deltas = []
        for i in range(len(self.layers) - 1, -1, -1):
            delta = layer_error * self.sigmoid_derivative(outputs[i + 1])
            deltas.append(delta)
            layer_error = delta.dot(self.layers[i].weights.T)
            grad = outputs[i].T.dot(delta)
            self.layers[i].weights += grad

    def forward_prop(self, inputs):
        cur_input = inputs
        outputs = []
        for i in range(len(self.layers)):
            output = self.sigmoid(dot(cur_input, self.layers[i].weights))
            outputs.append(output)
            cur_input = output
        return outputs

    # Use utils.classify() if data is categorical and does not need to ve normalized
    def classify(self, input_x):
        # call only after training
        if self.data_source == DataSource.WINE:
            for i in range(input_x.shape[1]):
                input_x[:, i] = (input_x[:, i] - self.data.means[i]) / self.data.standard_deviations[i]

        outputs = self.forward_prop(input_x)
        classified_output = utils.classify(outputs[-1])

        print(self.data.name, 'classify() result:\n', classified_output, '\nGiven input:\n', input_x)

    def train_wine_data(self):
        hidden_layer1_size = 10
        self.layers.append(NeuronLayer(hidden_layer1_size, wine_data.NUM_FEATURES))
        self.layers.append(NeuronLayer(wine_data.REGION_COUNT, hidden_layer1_size))
        self.train(1000)
        self.train_output = self.forward_prop(self.data.train_x)
        classified__tr_output = utils.classify(self.train_output[-1])
        num_tr_matches = 0
        num_te_matches = 0
        for i in range(self.data.train_x.shape[0]):
            if (classified__tr_output[i] == self.data.train_y[i]).all():
                num_tr_matches += 1

        self.test_output = self.forward_prop(self.data.test_x)
        classified_te_output = utils.classify(self.test_output[-1])
        for i in range(self.data.test_x.shape[0]):
            if (classified_te_output[i] == self.data.test_y[i]).all():
                num_te_matches += 1

        self.classify(array([[14.13, 1, 2.74, 15, 110, 2.05, 3, .3, 2, 5, 1, 3, 1000]]))
        print(self.data.name, 'Training set count:', classified__tr_output.shape[0], 'Number of matches:', num_tr_matches)
        print(self.data.name, 'Test set count:', classified_te_output.shape[0], 'Number of matches:', num_te_matches)
        print(self.data.name, 'Training set performance:', float(num_tr_matches) / float(classified__tr_output.shape[0]))
        print(self.data.name, 'Test set performance:', float(num_te_matches) / float(classified_te_output.shape[0]))

    def train_mushroom_data(self):
        hidden_layer1_size = 10
        self.layers.append(NeuronLayer(hidden_layer1_size, self.data.train_x.shape[1]))
        self.layers.append(NeuronLayer(mushrooms_data.CLASS_COUNT, hidden_layer1_size))
        self.train(1000)
        self.train_output = self.forward_prop(self.data.train_x)
        classified__tr_output = utils.classify(self.train_output[-1])
        num_tr_matches = 0
        num_te_matches = 0
        for i in range(self.data.train_x.shape[0]):
            if (classified__tr_output[i] == self.data.train_y[i]).all():
                num_tr_matches += 1

        self.test_output = self.forward_prop(self.data.test_x)
        classified_te_output = utils.classify(self.test_output[-1])
        for i in range(self.data.test_x.shape[0]):
            if (classified_te_output[i] == self.data.test_y[i]).all():
                num_te_matches += 1

        print(self.data.name, 'Training set count:', classified__tr_output.shape[0], 'Number of matches:',
              num_tr_matches)
        print(self.data.name, 'Test set count:', classified_te_output.shape[0], 'Number of matches:', num_te_matches)
        print(self.data.name, 'Training set performance:',
              float(num_tr_matches) / float(classified__tr_output.shape[0]))
        print(self.data.name, 'Test set performance:', float(num_te_matches) / float(classified_te_output.shape[0]))

    def train_flag_data(self):
        pass

    def print_weights(self):
        for i in range(len(self.layers)):
            print('Weights of layer', i, ':\n', self.layers[i].weights)
