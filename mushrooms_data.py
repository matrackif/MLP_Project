import pandas
import numpy
import utils
from sklearn.preprocessing import OneHotEncoder

"""
7. Attribute Information: (classes: edible=e, poisonous=p)
     1. cap-shape:                bell=b,conical=c,convex=x,flat=f,
                                  knobbed=k,sunken=s
     2. cap-surface:              fibrous=f,grooves=g,scaly=y,smooth=s
     3. cap-color:                brown=n,buff=b,cinnamon=c,gray=g,green=r,
                                  pink=p,purple=u,red=e,white=w,yellow=y
     4. bruises?:                 bruises=t,no=f
     5. odor:                     almond=a,anise=l,creosote=c,fishy=y,foul=f,
                                  musty=m,none=n,pungent=p,spicy=s
     6. gill-attachment:          attached=a,descending=d,free=f,notched=n
     7. gill-spacing:             close=c,crowded=w,distant=d
     8. gill-size:                broad=b,narrow=n
     9. gill-color:               black=k,brown=n,buff=b,chocolate=h,gray=g,
                                  green=r,orange=o,pink=p,purple=u,red=e,
                                  white=w,yellow=y
    10. stalk-shape:              enlarging=e,tapering=t
    11. stalk-root:               bulbous=b,club=c,cup=u,equal=e,
                                  rhizomorphs=z,rooted=r,missing=?
    12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
    13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
    14. stalk-color-above-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                  pink=p,red=e,white=w,yellow=y
    15. stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                  pink=p,red=e,white=w,yellow=y
    16. veil-type:                partial=p,universal=u
    17. veil-color:               brown=n,orange=o,white=w,yellow=y
    18. ring-number:              none=n,one=o,two=t
    19. ring-type:                cobwebby=c,evanescent=e,flaring=f,large=l,
                                  none=n,pendant=p,sheathing=s,zone=z
    20. spore-print-color:        black=k,brown=n,buff=b,chocolate=h,green=r,
                                  orange=o,purple=u,white=w,yellow=y
    21. population:               abundant=a,clustered=c,numerous=n,
                                  scattered=s,several=v,solitary=y
    22. habitat:                  grasses=g,leaves=l,meadows=m,paths=p,
                                  urban=u,waste=w,woods=d
                                  
    Attributes are first converted to numbers 0, 1, 2, ... Then one hot encoded.
    For example if cap shape = bell = b = 1 Then its y row would be [1, 0, 0, 0, 0, 0]
"""
CLASS_COUNT = 2
NUM_FEATURES = 22


class MushroomData:
    def __init__(self):
        self.file = 'mushrooms.txt'
        self.name = 'Mushroom'
        self.data = pandas.read_csv(self.file)
        self.values = self.data.values
        self.nrows = self.values.shape[0]
        self.ncols = self.values.shape[1]
        print('Mushroom data:\n', self.data.head(), 'Num Rows:', self.nrows, 'Num Cols:', self.ncols)
        self.train_percentage = 0.8
        self.train_count = int(self.train_percentage * self.nrows)

        self.train_x = numpy.empty((0, NUM_FEATURES))
        self.train_y = numpy.empty((0, 1))
        self.test_x = numpy.empty((0, NUM_FEATURES))
        self.test_y = numpy.empty((0, 1))

        # Classes (edible or poisonous)
        self.values[self.values[:, 0] == 'e', 0] = 0
        self.values[self.values[:, 0] == 'p', 0] = 1

        # Cap shape
        self.values[self.values[:, 1] == 'b', 1] = 0
        self.values[self.values[:, 1] == 'c', 1] = 1
        self.values[self.values[:, 1] == 'x', 1] = 2
        self.values[self.values[:, 1] == 'f', 1] = 3
        self.values[self.values[:, 1] == 'k', 1] = 4
        self.values[self.values[:, 1] == 's', 1] = 5

        # Cap surface
        self.values[self.values[:, 2] == 'f', 2] = 0
        self.values[self.values[:, 2] == 'g', 2] = 1
        self.values[self.values[:, 2] == 'y', 2] = 2
        self.values[self.values[:, 2] == 's', 2] = 3

        # Cap color
        self.values[self.values[:, 3] == 'n', 3] = 0
        self.values[self.values[:, 3] == 'b', 3] = 1
        self.values[self.values[:, 3] == 'c', 3] = 2
        self.values[self.values[:, 3] == 'g', 3] = 3
        self.values[self.values[:, 3] == 'r', 3] = 4
        self.values[self.values[:, 3] == 'p', 3] = 5
        self.values[self.values[:, 3] == 'u', 3] = 6
        self.values[self.values[:, 3] == 'e', 3] = 7
        self.values[self.values[:, 3] == 'w', 3] = 8
        self.values[self.values[:, 3] == 'y', 3] = 9

        # Bruises
        self.values[self.values[:, 4] == 't', 4] = 0
        self.values[self.values[:, 4] == 'f', 4] = 1

        # Odor
        self.values[self.values[:, 5] == 'a', 5] = 0
        self.values[self.values[:, 5] == 'l', 5] = 1
        self.values[self.values[:, 5] == 'c', 5] = 2
        self.values[self.values[:, 5] == 'y', 5] = 3
        self.values[self.values[:, 5] == 'f', 5] = 4
        self.values[self.values[:, 5] == 'm', 5] = 5
        self.values[self.values[:, 5] == 'n', 5] = 6
        self.values[self.values[:, 5] == 'p', 5] = 7
        self.values[self.values[:, 5] == 's', 5] = 8

        # gill-attachment
        self.values[self.values[:, 6] == 'a', 6] = 0
        self.values[self.values[:, 6] == 'd', 6] = 1
        self.values[self.values[:, 6] == 'f', 6] = 2
        self.values[self.values[:, 6] == 'n', 6] = 3

        # gill-spacing
        self.values[self.values[:, 7] == 'c', 7] = 0
        self.values[self.values[:, 7] == 'w', 7] = 1
        self.values[self.values[:, 7] == 'd', 7] = 2

        # gill-size
        self.values[self.values[:, 8] == 'b', 8] = 0
        self.values[self.values[:, 8] == 'n', 8] = 1

        # gill-color
        self.values[self.values[:, 9] == 'k', 9] = 0
        self.values[self.values[:, 9] == 'n', 9] = 1
        self.values[self.values[:, 9] == 'b', 9] = 2
        self.values[self.values[:, 9] == 'h', 9] = 3
        self.values[self.values[:, 9] == 'g', 9] = 4
        self.values[self.values[:, 9] == 'r', 9] = 5
        self.values[self.values[:, 9] == 'o', 9] = 6
        self.values[self.values[:, 9] == 'p', 9] = 7
        self.values[self.values[:, 9] == 'u', 9] = 8
        self.values[self.values[:, 9] == 'e', 9] = 9
        self.values[self.values[:, 9] == 'w', 9] = 10
        self.values[self.values[:, 9] == 'y', 9] = 11

        # stalk-shape
        self.values[self.values[:, 10] == 'e', 10] = 0
        self.values[self.values[:, 10] == 't', 10] = 1

        # stalk-root
        self.values[self.values[:, 11] == 'b', 11] = 0
        self.values[self.values[:, 11] == 'c', 11] = 1
        self.values[self.values[:, 11] == 'u', 11] = 2
        self.values[self.values[:, 11] == 'e', 11] = 3
        # assign rows with missing stalk root the "equal" attribute
        self.values[self.values[:, 11] == '?', 11] = 3
        self.values[self.values[:, 11] == 'z', 11] = 4
        self.values[self.values[:, 11] == 'r', 11] = 5

        # stalk-surface-above-ring
        self.values[self.values[:, 12] == 'f', 12] = 0
        self.values[self.values[:, 12] == 'y', 12] = 1
        self.values[self.values[:, 12] == 'k', 12] = 2
        self.values[self.values[:, 12] == 's', 12] = 3

        # stalk-surface-below-ring
        self.values[self.values[:, 13] == 'f', 13] = 0
        self.values[self.values[:, 13] == 'y', 13] = 1
        self.values[self.values[:, 13] == 'k', 13] = 2
        self.values[self.values[:, 13] == 's', 13] = 3

        # stalk-color-above-ring
        self.values[self.values[:, 14] == 'n', 14] = 0
        self.values[self.values[:, 14] == 'b', 14] = 1
        self.values[self.values[:, 14] == 'c', 14] = 2
        self.values[self.values[:, 14] == 'g', 14] = 3
        self.values[self.values[:, 14] == 'o', 14] = 4
        self.values[self.values[:, 14] == 'p', 14] = 5
        self.values[self.values[:, 14] == 'e', 14] = 6
        self.values[self.values[:, 14] == 'w', 14] = 7
        self.values[self.values[:, 14] == 'y', 14] = 8

        # stalk-color-below-ring
        self.values[self.values[:, 15] == 'n', 15] = 0
        self.values[self.values[:, 15] == 'b', 15] = 1
        self.values[self.values[:, 15] == 'c', 15] = 2
        self.values[self.values[:, 15] == 'g', 15] = 3
        self.values[self.values[:, 15] == 'o', 15] = 4
        self.values[self.values[:, 15] == 'p', 15] = 5
        self.values[self.values[:, 15] == 'e', 15] = 6
        self.values[self.values[:, 15] == 'w', 15] = 7
        self.values[self.values[:, 15] == 'y', 15] = 8

        # veil type
        self.values[self.values[:, 16] == 'p', 16] = 0
        self.values[self.values[:, 16] == 'u', 16] = 1

        # veil-color
        self.values[self.values[:, 17] == 'n', 17] = 0
        self.values[self.values[:, 17] == 'o', 17] = 1
        self.values[self.values[:, 17] == 'w', 17] = 2
        self.values[self.values[:, 17] == 'y', 17] = 3

        # ring-number
        self.values[self.values[:, 18] == 'n', 18] = 0
        self.values[self.values[:, 18] == 'o', 18] = 1
        self.values[self.values[:, 18] == 't', 18] = 2

        # ring-type
        self.values[self.values[:, 19] == 'c', 19] = 0
        self.values[self.values[:, 19] == 'e', 19] = 1
        self.values[self.values[:, 19] == 'f', 19] = 2
        self.values[self.values[:, 19] == 'l', 19] = 3
        self.values[self.values[:, 19] == 'n', 19] = 4
        self.values[self.values[:, 19] == 'p', 19] = 5
        self.values[self.values[:, 19] == 's', 19] = 6
        self.values[self.values[:, 19] == 'z', 19] = 7

        # spore-print-color
        self.values[self.values[:, 20] == 'k', 20] = 0
        self.values[self.values[:, 20] == 'n', 20] = 1
        self.values[self.values[:, 20] == 'b', 20] = 2
        self.values[self.values[:, 20] == 'h', 20] = 3
        self.values[self.values[:, 20] == 'r', 20] = 4
        self.values[self.values[:, 20] == 'o', 20] = 5
        self.values[self.values[:, 20] == 'u', 20] = 6
        self.values[self.values[:, 20] == 'w', 20] = 7
        self.values[self.values[:, 20] == 'y', 20] = 8

        # population
        self.values[self.values[:, 21] == 'a', 21] = 0
        self.values[self.values[:, 21] == 'c', 21] = 1
        self.values[self.values[:, 21] == 'n', 21] = 2
        self.values[self.values[:, 21] == 's', 21] = 3
        self.values[self.values[:, 21] == 'v', 21] = 4
        self.values[self.values[:, 21] == 'y', 21] = 5

        # habitat
        self.values[self.values[:, 22] == 'g', 22] = 0
        self.values[self.values[:, 22] == 'l', 22] = 1
        self.values[self.values[:, 22] == 'm', 22] = 2
        self.values[self.values[:, 22] == 'p', 22] = 3
        self.values[self.values[:, 22] == 'u', 22] = 4
        self.values[self.values[:, 22] == 'w', 22] = 5
        self.values[self.values[:, 22] == 'd', 22] = 6

        print('Mushrooms values:', self.values)

        for i in range(CLASS_COUNT):
            tmp = self.values[self.values[:, 0] == i]
            tr_count = int(self.train_percentage * tmp.shape[0])
            tr_x = tmp[:tr_count, 1:]
            tr_y = tmp[:tr_count, 0].reshape(-1, 1)
            te_x = tmp[tr_count:, 1:]
            te_y = tmp[tr_count:, 0].reshape(-1, 1)
            self.train_x = numpy.append(self.train_x, tr_x, axis=0)
            self.train_y = numpy.append(self.train_y, tr_y, axis=0)
            self.test_x = numpy.append(self.test_x, te_x, axis=0)
            self.test_y = numpy.append(self.test_y, te_y, axis=0)

        print('train_x rows:', self.train_x.shape[0], 'train_x cols:', self.train_x.shape[1])
        print('train_y rows:', self.train_y.shape[0], 'train_y cols:', self.train_y.shape[1])
        print('test_x rows:', self.test_x.shape[0], 'test_x cols:', self.test_x.shape[1])
        print('test_y rows:', self.test_y.shape[0], 'test_y cols:', self.test_y.shape[1])
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
        self.encoder_y = OneHotEncoder()
        self.encoder_y.fit(self.train_y)
        self.train_y = self.encoder_y.transform(self.train_y).toarray()
        self.test_y = self.encoder_y.transform(self.test_y).toarray()

        self.encoder_x = OneHotEncoder()
        self.encoder_x.fit(numpy.append(self.train_x, self.test_x, axis=0))
        self.train_x = self.encoder_x.transform(self.train_x).toarray()
        self.test_x = self.encoder_x.transform(self.test_x).toarray()

        print('train_x rows:', self.train_x.shape[0], 'train_x cols:', self.train_x.shape[1])
        print('train_y rows:', self.train_y.shape[0], 'train_y cols:', self.train_y.shape[1])
        print('test_x rows:', self.test_x.shape[0], 'test_x cols:', self.test_x.shape[1])
        print('test_y rows:', self.test_y.shape[0], 'test_y cols:', self.test_y.shape[1])
        print('\ntrain_x:\n', self.train_x[0])

        # print('\ntrain_y:\n', self.train_y)
        # print('\ntest_x:\n', self.test_x)
        # print('\ntest_y:\n', self.test_y)