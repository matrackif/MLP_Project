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
    arg_parser.add_argument('-n', type=int, default=100, help='Number of training iterations (of back propagation)')
    args = vars(arg_parser.parse_args())
    print('Command line args:', args)
    if args['w']:
        neural_network = NeuralNetwork(data_source=DataSource.WINE, num_iters=args['n'])
    if args['m']:
        neural_network = NeuralNetwork(data_source=DataSource.MUSHROOMS, num_iters=args['n'])
    if args['f']:
        neural_network = NeuralNetwork(data_source=DataSource.FLAGS, num_iters=args['n'])