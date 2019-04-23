import numpy as np
from backprop import Backprop

def booltrain(target):
    pair = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
    per = Backprop(2, 1, 3)
    per.train(pair,target, t=10000)
    for i in range(4):
        print(str(pair[i].astype('bool')) + ' >>', per.test(pair[i], target[i]))
        print()

target_xor = ([1, 0, 0, 1])

print("XOR Function")
booltrain(target_xor)
print()