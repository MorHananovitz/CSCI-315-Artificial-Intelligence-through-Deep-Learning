import numpy as np
from backprop import Backprop
from srn import SRN

def xorseq(n):
    inputs = []
    for i in range(n):
        randbol1 = np.random.randint(2, size=1)
        randbol2 = np.random.randint(2, size=1)
        target = randbol1^randbol2
        inputs.append(randbol1)
        inputs.append(randbol2)
        inputs.append(target)
    return inputs

def shift(inputs):
    shiftleft = inputs[1:]
    shiftleft.append([0])
    return shiftleft

#Backprop
bp = Backprop(1, 1, 2) #1 input, 1 outputs, 2 hidden layers
inputs = xorseq(1000)
targets = shift(inputs)
bp.train(inputs, targets, eta=0.5, t=600)

#SRN
srn = SRN(1, 1, 2) #1 input, 1 outputs, 2 hidden layers
srn.train(inputs, targets, eta=0.5, t=600)