import numpy


def normalize(input: numpy.array):
    ncols = input.shape[1]
    means = numpy.zeros((ncols,))
    standard_deviations = numpy.zeros((ncols,))
    for i in range(input.shape[1]):
        means[i] = numpy.mean(input[:, i])
        standard_deviations[i] = numpy.std(input[:, i])
        print('For col', i, 'mean is', means[i], 'and std is', standard_deviations[i])
        input[:, i] = (input[:, i] - means[i]) / standard_deviations[i]


def classify(result: numpy.array):
    tmp = numpy.zeros_like(result)
    tmp[range(len(result)), result.argmax(axis=1)] = 1
    return tmp
