import pandas  as pd
import numpy as np
import random



def read_train_data():

    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    index = np.arange(0, len(x_train))
    random.shuffle(index)
    x_train_shuffle = x_train[index]
    y_train_shuffle = y_train[index]
    print(x_train_shuffle.shape)
    print(y_train_shuffle.shape)
    return x_train_shuffle, y_train_shuffle

def read_test_data():

    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')
    index = np.arange(0, len(x_test))
    random.shuffle(index)
    x_test_shuffle = x_test[index]
    y_test_shuffle = y_test[index]

    return x_test_shuffle, y_test_shuffle



x_test, y_test = read_test_data()
print('1')