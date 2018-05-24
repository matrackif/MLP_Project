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
        print('self.values:\n', self.values)

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

        self.train_x = numpy.empty((0, self.train_x_reg1.shape[1]))
        self.train_x = numpy.append(self.train_x, self.train_x_reg1, axis=0)
        self.train_x = numpy.append(self.train_x, self.train_x_reg2, axis=0)
        self.train_x = numpy.append(self.train_x, self.train_x_reg3, axis=0)

        self.train_y = numpy.empty((0, 1))
        self.train_y = numpy.append(self.train_y, self.train_y_reg1, axis=0)
        self.train_y = numpy.append(self.train_y, self.train_y_reg2, axis=0)
        self.train_y = numpy.append(self.train_y, self.train_y_reg3, axis=0)

        self.test_x = numpy.empty((0, self.test_x_reg1.shape[1]))
        self.test_x = numpy.append(self.test_x, self.test_x_reg1, axis=0)
        self.test_x = numpy.append(self.test_x, self.test_x_reg2, axis=0)
        self.test_x = numpy.append(self.test_x, self.test_x_reg3, axis=0)

        self.test_y = numpy.empty((0, 1))
        self.test_y = numpy.append(self.test_y, self.test_y_reg1, axis=0)
        self.test_y = numpy.append(self.test_y, self.test_y_reg2, axis=0)
        self.test_y = numpy.append(self.test_y, self.test_y_reg3, axis=0)

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