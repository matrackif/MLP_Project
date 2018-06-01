import pandas
import utils
import numpy
from sklearn.preprocessing import OneHotEncoder


class WineData:
    def __init__(self):
        self.file = 'wine.txt'
        self.data = pandas.read_csv(self.file)
        self.values = self.data.values
        self.nrows = self.values.shape[0]
        self.ncols = self.values.shape[1]
        print('Data:\n', self.data.head(), 'Num Rows:', self.nrows, 'Num Cols:', self.ncols)
        self.train_percentage = 0.8
        self.train_count = int(self.train_percentage * self.nrows)
        self.means, self.standard_deviations = utils.normalize(self.values[:, 1:])
        # print('self.values:\n', self.values)
        REGION_COUNT = 3
        NUM_FEATURES = 13
        self.train_x = numpy.empty((0, NUM_FEATURES))
        self.train_y = numpy.empty((0, 1))
        self.test_x = numpy.empty((0, NUM_FEATURES))
        self.test_y = numpy.empty((0, 1))

        for i in range(REGION_COUNT):
            reg = self.values[self.values[:, 0] == i + 1]
            tr_count = int(self.train_percentage * reg.shape[0])
            tr_x = reg[:tr_count, 1:]
            tr_y = reg[:tr_count, 0].reshape(-1, 1)
            te_x = reg[tr_count:, 1:]
            te_y = reg[tr_count:, 0].reshape(-1, 1)
            self.train_x = numpy.append(self.train_x, tr_x, axis=0)
            self.train_y = numpy.append(self.train_y, tr_y, axis=0)
            self.test_x = numpy.append(self.test_x, te_x, axis=0)
            self.test_y = numpy.append(self.test_y, te_y, axis=0)

        self.train_count = self.train_x.shape[0]
        self.test_count = self.test_x.shape[0]
        self.feature_count = self.train_x.shape[1]
        self.output_count = self.train_y.shape[1]

        """
        print('train_x rows:', self.train_x.shape[0], 'train_x cols:', self.train_x.shape[1])
        print('train_y rows:', self.train_y.shape[0], 'train_y cols:', self.train_y.shape[1])
        print('test_x rows:', self.test_x.shape[0], 'test_x cols:', self.test_x.shape[1])
        print('test_y rows:', self.test_y.shape[0], 'test_y cols:', self.test_y.shape[1])

        print('\ntrain_x:\n', self.train_x)
        print('\ntrain_y:\n', self.train_y)
        print('\ntest_x:\n', self.test_x)
        print('\ntest_y:\n', self.test_y)
        """

        # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        enc = OneHotEncoder()
        enc.fit(self.train_y)
        self.train_y = enc.transform(self.train_y).toarray()
        self.test_y = enc.transform(self.test_y).toarray()