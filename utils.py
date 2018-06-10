import numpy
from typing import List


def normalize(input: numpy.array, indices: List = None):
    means = None
    standard_deviations = None
    if indices is None:
        ncols = input.shape[1]
        means = numpy.zeros((ncols,))
        standard_deviations = numpy.zeros((ncols,))
        for i in range(input.shape[1]):
            means[i] = numpy.mean(input[:, i])
            standard_deviations[i] = numpy.std(input[:, i])
            print('For col', i, 'mean is', means[i], 'and std is', standard_deviations[i])
            input[:, i] = (input[:, i] - means[i]) / standard_deviations[i]
    else:
        ncols = len(indices)
        means = numpy.zeros((ncols,))
        standard_deviations = numpy.zeros((ncols,))
        k = 0
        for i in indices:
            means[k] = numpy.mean(input[:, i])
            standard_deviations[k] = numpy.std(input[:, i])
            print('For col', i, 'mean is', means[k], 'and std is', standard_deviations[k])
            input[:, i] = (input[:, i] - means[k]) / standard_deviations[k]
            k += 1
    return means, standard_deviations


def classify(result: numpy.array):
    tmp = numpy.zeros_like(result)
    tmp[range(len(result)), result.argmax(axis=1)] = 1
    return tmp
