import numpy as np
from backpropfast import Backprop
import pickle

def booltrain(target):
    pair = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
    back = Backprop(n= 2, m= 1, h = 6, part = 1)
    eta, t, h, mew, RMSerr = back.train(pair,target, t=10000)

    for i in range(4):
        print(str(pair[i].astype('bool')) + ' >>', back.test(pair[i], target[i]))
        print()
    print("XOR Function Results with eta=%.1f %d Iterations and %d Hidden Units" % (eta, t, h))

target_xor = np.array([1, 0, 0, 1]).reshape(-1,1)
booltrain(target_xor)
