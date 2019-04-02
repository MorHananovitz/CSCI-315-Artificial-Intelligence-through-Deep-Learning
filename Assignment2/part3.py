import numpy as np
import itertools as it
from perceptron import Perceptron
from digit import dataread

input_train, _ = dataread('digits_train.txt')
input_test, _ = dataread('digits_test.txt')

targets = np.zeros((2500,1))
targets[500:750] = 1

per = Perceptron(196, 1)
per.train(input_train, targets)

falneg_counter = []
for i in range(500, 750):
    falneg_counter.append(per.test(input_test[i]))
false_neg_rate = (250-np.sum(falneg_counter))/250*100

falpos_counter = []
for i in it.chain(range(0,500), range(750,2500)):
    falpos_counter.append(per.test(input_test[i]))
false_pos_rate = np.sum(falpos_counter)/2250*100

print()
print('False Positive Rate is %d %%' % false_pos_rate)
print('False Negative Rate is %d %%' % false_neg_rate)