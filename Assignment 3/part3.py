import numpy as np
import itertools as it
import pandas as pd
from backpropfast import Backprop
import re
import sys


def pretty_print(df,precision):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None,'precision', precision):  # more options can be specified also
        print(df)

def Vector2num(target):
    return np.where(target > 0)[0]

def dataread(file_name):
    list_input_vector = []
    list_target = []
    data = pd.read_csv(file_name, header=None)
    for i in range(len(data)//16):
        list_data=[]
        for col in data.values[16*i + 1: 16*i + 15]:
            list_data.append([int(x) for x in re.findall('\d+', col[0])])
        vector_build = np.concatenate(list_data)

        # Creating the target (str 2 num)
        target = np.array([int(x) for x in re.findall('\d+', data.values[16*i + 15][0])])

        # Convert vector to target
        target_build = Vector2num(target)

        list_input_vector.append(vector_build)
        list_target.append(target_build)
    return list_input_vector, list_target

input_train, _ = dataread('digits_train.txt')
input_test, _ = dataread('digits_test.txt')

targets = np.zeros((2500,10))
targets[:250, 0] = 1
targets[250:500, 1] = 1
targets[500:750, 2] = 1
targets[750:1000 ,3] = 1
targets[1000:1250, 4] = 1
targets[1250:1500, 5] = 1
targets[1500:1750, 6] = 1
targets[1750:2000, 7] = 1
targets[2000:2250, 8] = 1
targets[2250:, 9] = 1

NN = Backprop(n = 196, m = 10, h = 20, part = 3)
eta, t, h, mew, RMSerr, WIH_00, WHO_00  = NN.train(np.array(input_train), np.array(targets), eta=0.5, t=5000, n_batch = 500)

confusion_matrix = []

for i in range(10):
    success = np.zeros(10)
    for j in range(i*250, (i+1)*250):
        NN.test(input_test[j], targets[i])
        success += NN.test(input_test[j], targets[i])
    confusion_matrix.append(success)

print()
print("10 Digit Classifier Results with eta=%.1f %d Iterations and %d Hidden Units" % (eta, t, h))
print()
A = np.array(confusion_matrix)
title = [_ for _ in '0123456789']
B = pd.DataFrame(A, index= title, columns= title)
B.to_csv('B.csv', index=True, header=True, sep=' ')
pretty_print(B,2)

name_file_WHO = "Part_%d_WHO_eta=%.1f_t=%d_h=%d.wgt" % (3, eta, t , h)
name_file_WIH = "Part_%d_WIH_eta=%.1f_t=%d_h=%d.wgt" % (3, eta, t , h)

see_WHO = NN.load_mine(name_file_WHO)
see_WIH = NN.load_mine(name_file_WIH)