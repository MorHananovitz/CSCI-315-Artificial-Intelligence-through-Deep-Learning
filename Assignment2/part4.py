import numpy as np
import itertools as it
from perceptron import Perceptron
from digit import dataread

input_train, _ = dataread('digits_train.txt')
input_test, _ = dataread('digits_test.txt')

A = input_train[0:250]
B = input_train[2000:2250]

input_vec_train=np.vstack((A, B))

C = input_test[0:250]
D = input_test[2000:2250]

input_vec_test =np.vstack((C, D))

targets = np.zeros((500,2))
targets[:250,0] = 1
targets[250:,1] = 1

per = Perceptron(196, 2)
per.train(input_vec_train, targets)

zero_counter = []
for i in range(0, 250):
    zero_counter.append(per.test(input_vec_test[i]))

zero_success = np.sum(zero_counter, axis=0)[0]/250*100

eight_counter = []
for i in range(250, 500):
    eight_counter.append(per.test(input_vec_test[i]))
eight_success = np.sum(eight_counter, axis=0)[1] / 250 * 100

print()
print('0 Classified correct is: %d %%' % zero_success)
print('8 Classified correct is: %d %%' % eight_success)