from numpy import exp, array, random, dot, extract
import calendar, time
from enum import Enum
import wine_data, mushrooms_data, flags_data
from matplotlib import pyplot as plt
import utils


class DataSource(Enum):
    WINE = 1
    MUSHROOMS = 2
    FLAGS = 3


class NeuronLayer:
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        random.seed(calendar.timegm(time.gmtime()))
        self.weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork:
    def __init__(self, data_source: DataSource, num_iters: int, num_hidden_layers: int, hidden_layer_size: int, num_test_intervals: int, run_tests: bool):
        self.input_layer = None
        self.output_layer = None
        self.train_output = None
        self.test_output = None
        self.data_source = data_source
        self.data = None
        self.data_name = ''
        self.layers = []
        self.num_iters = num_iters
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.num_test_intervals = num_test_intervals
        self.run_tests = run_tests

        if self.data_source == DataSource.WINE:
            self.data = wine_data.WineData()
        elif self.data_source == DataSource.MUSHROOMS:
            self.data = mushrooms_data.MushroomData()
        elif self.data_source == DataSource.FLAGS:
            self.data = flags_data.FlagData()
        if self.run_tests:
            self.test_number_of_training_iterations()
        else:
            self.run(self.num_iters)

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))
        # return .5 * (1 + numpy.tanh(.5 * x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def train(self, num_iters:int = 0):
        for iteration in range(num_iters):
            self.back_prop()

    def back_prop(self):
        learning_rate = 0.5
        outputs = [self.data.train_x]
        ret = self.forward_prop(self.data.train_x)
        for output in ret:
            outputs.append(output)
        # print('len of layers:', len(self.layers))
        # print('outputs[-1]:', outputs[-1])
        layer_error = self.data.train_y - outputs[-1]
        # print('MSE layer_error:', (layer_error ** 2).sum())
        deltas = []
        for i in range(len(self.layers) - 1, -1, -1):
            delta = layer_error * self.sigmoid_derivative(outputs[i + 1])
            deltas.append(delta)
            layer_error = delta.dot(self.layers[i].weights.T)
            grad = outputs[i].T.dot(delta)
            self.layers[i].weights += grad * learning_rate

    def forward_prop(self, inputs):
        cur_input = inputs
        outputs = []
        for i in range(len(self.layers)):
            output = self.sigmoid(dot(cur_input, self.layers[i].weights))
            outputs.append(output)
            cur_input = output
        return outputs

    def classify(self, input_x):
        # call only after training
        if self.data_source == DataSource.WINE:
            for i in range(input_x.shape[1]):
                input_x[:, i] = (input_x[:, i] - self.data.means[i]) / self.data.standard_deviations[i]

        outputs = self.forward_prop(input_x)
        print('len(outputs):', len(outputs))
        classified_output = utils.classify(outputs[-1])
        print(self.data.name, 'classify() result:\n', classified_output, '\nGiven input:\n', input_x)

    def run(self, num_iters: int = 0):
        self.add_layers()
        self.train(num_iters)
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
        train_perf = float(num_tr_matches) / float(classified__tr_output.shape[0]) * 100
        test_perf = float(num_te_matches) / float(classified_te_output.shape[0]) * 100
        print(self.data.name, 'Training set count:', classified__tr_output.shape[0], 'Number of matches:', num_tr_matches)
        print(self.data.name, 'Test set count:', classified_te_output.shape[0], 'Number of matches:', num_te_matches)
        print(self.data.name, 'Training set performance:', train_perf, '%')
        print(self.data.name, 'Test set performance:', test_perf, '%')
        return train_perf, test_perf

    def add_layers(self):
        self.layers = []
        if self.num_hidden_layers == 0:
            self.layers.append(NeuronLayer(self.data.num_classes, self.data.train_x.shape[1]))
            return
        elif self.num_hidden_layers == 1:
            self.layers.append(NeuronLayer(self.hidden_layer_size, self.data.train_x.shape[1]))
            self.layers.append(NeuronLayer(self.data.num_classes, self.hidden_layer_size))
            return
        else:
            # size = self.hidden_layer_size
            for i in range(self.num_hidden_layers):
                # size = int(size / 2)
                if i == 0:
                    self.layers.append(NeuronLayer(self.hidden_layer_size, self.data.train_x.shape[1]))
                elif i == self.num_hidden_layers - 1:
                    self.layers.append(NeuronLayer(self.data.num_classes, self.hidden_layer_size))
                else:
                    self.layers.append(NeuronLayer(self.hidden_layer_size, self.hidden_layer_size))

    def print_weights(self):
        for i in range(len(self.layers)):
            print('Weights of layer', i, ':\n', self.layers[i].weights)

    def test_number_of_training_iterations(self):
        iters_list = []
        interval_length = self.num_iters / self.num_test_intervals
        k = int(interval_length)
        while k <= self.num_iters:
            iters_list.append(int(k))
            k += interval_length
        if len(iters_list) != self.num_test_intervals:
            iters_list.append(self.num_iters)
        print('Will test NN error with following number of iterations:', iters_list)

        tr_performances = []
        te_performances = []

        for n in iters_list:
            tr_p, te_p = self.run(n)
            tr_performances.append(tr_p)
            te_performances.append(te_p)

        plt.figure(0)
        plt.title('Training performance with ' + str(self.num_hidden_layers) + ' hidden layer(s) on the ' + self.data.name + ' data set. \nHidden layer size: ' + str(self.hidden_layer_size))
        plt.xlabel('Number of training iterations')
        plt.ylabel('Percentage of correct predictions')
        tr_graph, = plt.plot(iters_list, tr_performances, label='Training set')
        te_graph, = plt.plot(iters_list, te_performances, label='Test set')
        plt.legend(handles=[tr_graph, te_graph])
        plt.show()
