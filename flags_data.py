import pandas
import numpy
import utils
from sklearn.preprocessing import OneHotEncoder
NUM_FEATURES = 28  # 30 columns - name column - class column
NUM_CLASSES = 8  # 8 religions


class FlagData:
    def __init__(self):
        self.name = 'Flag'
        self.file = 'flags.txt'
        self.data = pandas.read_csv(self.file)
        self.values = self.data.values
        self.nrows = self.values.shape[0]
        self.ncols = self.values.shape[1]
        self.num_classes = 8
        print('Wine Data:\n', self.data.head(), 'Num Rows:', self.nrows, 'Num Cols:', self.ncols)
        self.train_percentage = 0.9
        self.train_count = int(self.train_percentage * self.nrows)
        self.num_classes = 8
        feature_indices = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                           27, 28, 29]
        category_indices = [1, 2, 5, 10, 11, 12, 13, 14, 15, 16, 17, 23, 24, 25, 26, 27, 28, 29]
        numerical_indices = list(set(feature_indices).symmetric_difference(set(category_indices)))
        class_idx = 6

        utils.normalize(self.values, indices=numerical_indices)
        # mainhue color, column index 17
        self.values[self.values[:, 17] == 'red', 17] = 0
        self.values[self.values[:, 17] == 'green', 17] = 1
        self.values[self.values[:, 17] == 'blue', 17] = 2
        self.values[self.values[:, 17] == 'blue', 17] = 3
        self.values[self.values[:, 17] == 'gold', 17] = 4
        self.values[self.values[:, 17] == 'white', 17] = 5
        self.values[self.values[:, 17] == 'black', 17] = 6
        self.values[self.values[:, 17] == 'orange', 17] = 7
        self.values[self.values[:, 17] == 'brown', 17] = 8

        # top-left color
        self.values[self.values[:, 28] == 'red', 28] = 0
        self.values[self.values[:, 28] == 'green', 28] = 1
        self.values[self.values[:, 28] == 'blue', 28] = 2
        self.values[self.values[:, 28] == 'blue', 28] = 3
        self.values[self.values[:, 28] == 'gold', 28] = 4
        self.values[self.values[:, 28] == 'white', 28] = 5
        self.values[self.values[:, 28] == 'black', 28] = 6
        self.values[self.values[:, 28] == 'orange', 28] = 7
        self.values[self.values[:, 28] == 'brown', 28] = 8

        # bottom-right color
        self.values[self.values[:, 29] == 'red', 29] = 0
        self.values[self.values[:, 29] == 'green', 29] = 1
        self.values[self.values[:, 29] == 'blue', 29] = 2
        self.values[self.values[:, 29] == 'blue', 29] = 3
        self.values[self.values[:, 29] == 'gold', 29] = 4
        self.values[self.values[:, 29] == 'white', 29] = 5
        self.values[self.values[:, 29] == 'black', 29] = 6
        self.values[self.values[:, 29] == 'orange', 29] = 7
        self.values[self.values[:, 29] == 'brown', 29] = 8

        # print('numerical_indices:', numerical_indices)
        # print('self.values numerical features: ', self.values[:, numerical_indices])

        self.train_x = numpy.empty((0, 30))
        self.train_y = numpy.empty((0, 1))
        self.test_x = numpy.empty((0, 30))
        self.test_y = numpy.empty((0, 1))
        # Sort data by religion and divide into training/test sets
        for i in range(NUM_CLASSES):
            current_religion = self.values[self.values[:, class_idx] == i]
            tr_count = int(self.train_percentage * current_religion.shape[0])
            tr_x = current_religion[:tr_count]
            tr_y = current_religion[:tr_count, class_idx].reshape(-1, 1)
            te_x = current_religion[tr_count:]
            te_y = current_religion[tr_count:, class_idx].reshape(-1, 1)
            self.train_x = numpy.append(self.train_x, tr_x, axis=0)
            self.train_y = numpy.append(self.train_y, tr_y, axis=0)
            self.test_x = numpy.append(self.test_x, te_x, axis=0)
            self.test_y = numpy.append(self.test_y, te_y, axis=0)

        # Reorganize data by splitting numerical and categorical columns
        # (maybe could be done in loop above but it would be complicated)
        train_x_cat = self.train_x[:, category_indices]
        train_x_num = self.train_x[:, numerical_indices]
        self.train_countries = self.train_x[:, 0]

        test_x_cat = self.test_x[:, category_indices]
        test_x_num = self.test_x[:, numerical_indices]
        self.test_countries = self.test_x[:, 0]

        # https://stackoverflow.com/questions/34089906/sklearn-mask-for-onehotencoder-does-not-work
        # OneHotEncoder needs all numerical values even if categorical_features mask is specified (lol)
        enc = OneHotEncoder()
        enc.fit(train_x_cat)
        train_x_cat = enc.transform(train_x_cat).toarray()
        # print('train_x_cat:\n', train_x_cat[0])
        test_x_cat = enc.transform(test_x_cat).toarray()
        # print('train_x_cat.shape:', train_x_cat.shape, '\ntrain_x_num.shape:', train_x_num.shape)

        self.train_x = numpy.column_stack((train_x_num, train_x_cat))
        self.test_x = numpy.column_stack((test_x_num, test_x_cat))

        enc.fit(self.train_y)
        self.train_y = enc.transform(self.train_y).toarray()
        self.test_y = enc.transform(self.test_y).toarray()

        self.train_x = self.train_x.astype(numpy.float)
        self.test_x = self.test_x.astype(numpy.float)
        self.train_y = self.train_y.astype(numpy.float)
        self.test_y = self.test_y.astype(numpy.float)
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
        print('train_x rows:', self.train_x.shape[0], 'train_x cols:', self.train_x.shape[1])
        print('train_y rows:', self.train_y.shape[0], 'train_y cols:', self.train_y.shape[1])
        print('test_x rows:', self.test_x.shape[0], 'test_x cols:', self.test_x.shape[1])
        print('test_y rows:', self.test_y.shape[0], 'test_y cols:', self.test_y.shape[1])
