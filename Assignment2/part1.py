import numpy as np
from perceptron import Perceptron

def booltrain(target):
    pair = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
    per = Perceptron(2,1)
    per.train(pair,target, t=10)
    for i in range(4):
        print(str(pair[i].astype('bool')) + ' >>', per.test(pair[i]))
        print()

target_or = ([1, 1, 1, 0])
target_and = ([1, 0, 0, 0])

print("OR Function")
booltrain(target_or)
print()
print("AND Function")
booltrain(target_and)