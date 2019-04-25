import numpy as np
import itertools as it
import pandas as pd
from backpropfast import Backprop
import re

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

targets = np.zeros((2500,1))
targets[500:750] = 1

NN = Backprop(n = 196, m = 1, h = 20, part = 2)
eta, t, h, mew, RMSerr = NN.train(np.array(input_train), np.array(targets), eta=0.1, t=10000, n_batch = 1000)

falneg_counter = []
for i in range(500, 750):
    falneg_counter.append(NN.test(input_test[i], targets[i]))
false_neg_rate = (250-np.sum(falneg_counter))/250*100

falpos_counter = []
for i in it.chain(range(0,500), range(750,2500)):
    falpos_counter.append(NN.test(input_test[i], targets[i]))
false_pos_rate = np.sum(falpos_counter)/2250*100

print("2-Not 2 Results with eta=%.1f %d Iterations and %d Hidden Units" % (eta, t, h))

name_file_WHO = "Part_%d_WHO_eta=%.1f_t=%d_h=%d.wgt" % (2, eta, t , h)
name_file_WIH = "Part_%d_WIH_eta=%.1f_t=%d_h=%d.wgt" % (2, eta, t , h)

see_WHO = NN.load_mine(name_file_WHO)
see_WIH = NN.load_mine(name_file_WIH)

print()
print('False Positive Rate is %d %%' % false_pos_rate)
print('False Negative Rate is %d %%' % false_neg_rate)