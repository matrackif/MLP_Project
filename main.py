from numpy import exp, array, random, dot
from nn import NeuralNetwork, NeuronLayer
from nn import DataSource
from argparse import ArgumentParser

if __name__ == "__main__":

    # Seed the random number generator
    random.seed(1)
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-w', action='store_true', default=False, help='Train wine data model and print results')
    arg_parser.add_argument('-m', action='store_true', default=False,
                            help='Train mushroom data model and print results')
    arg_parser.add_argument('-f', action='store_true', default=False, help='Train flag data model and print results')
    arg_parser.add_argument('-t', action='store_true', default=False, help='Run tests and plot')
    arg_parser.add_argument('-n', type=int, default=100, help='Number of training iterations (of back propagation)')
    arg_parser.add_argument('-l', type=int, default=1, help='Number of hidden layers')
    arg_parser.add_argument('-s', type=int, default=30, help='Hidden layer size (number of neurons)')
    arg_parser.add_argument('-i', type=int, default=5, help='Number of test intervals. If num of training iterations is 100 and this argument is 4, then we will test error with 25, 50, 75, 100 iters')
    args = vars(arg_parser.parse_args())
    print('Command line args:', args)
    if args['w']:
        neural_network = NeuralNetwork(data_source=DataSource.WINE, num_iters=args['n'], num_hidden_layers=args['l'], hidden_layer_size=args['s'], num_test_intervals=args['i'], run_tests=args['t'])
    if args['m']:
        neural_network = NeuralNetwork(data_source=DataSource.MUSHROOMS, num_iters=args['n'], num_hidden_layers=args['l'], hidden_layer_size=args['s'], num_test_intervals=args['i'], run_tests=args['t'])
    if args['f']:
        neural_network = NeuralNetwork(data_source=DataSource.FLAGS, num_iters=args['n'], num_hidden_layers=args['l'], hidden_layer_size=args['s'], num_test_intervals=args['i'], run_tests=args['t'])