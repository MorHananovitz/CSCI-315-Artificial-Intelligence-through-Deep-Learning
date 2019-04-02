import numpy as np
import itertools as it
from perceptron import Perceptron
from digit import dataread

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

per = Perceptron(196, 10)
per.train(input_train, targets)

confusion_matrix = []

for i in range(10):
    success = np.zeros(10)
    for j in range(i*250, (i+1)*250):
        per.test(input_test[j])
        success += per.test(input_test[j])
    confusion_matrix.append(success)

print()
print(np.array(confusion_matrix))