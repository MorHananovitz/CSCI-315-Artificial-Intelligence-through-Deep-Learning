import numpy as np
from backprop import Backprop
from srn import SRN
import math



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

def xorerr(targets, inputs):
    Obp = bp.test(inputs)
    xorerr_bp = 0.0
    xorerr_srn = 0.0

    for count in range(len(targets)):
        if count % 3 == 1:
            xorerr_bp += (Obp[count]-targets[count]) ** 2
            xorerr_srn +=(srn.test(inputs[count])- targets[count]) ** 2

    #print(xorerr_bp*0.001)
    #print(xorerr_srn*0.001)

    return xorerr_bp*0.001, xorerr_srn*0.001

print("This is Backprop")
bp = Backprop(1, 1, 2) #1 input, 1 outputs, 2 hidden layers
inputs = xorseq(1000)
targets = shift(inputs)
eta, t, hiddenunits, RMSerr=bp.train(inputs, targets, eta=0.5, t=100)

print()
print("This is SRN")
srn = SRN(1, 1, 2) #1 input, 1 outputs, 2 hidden layers
srn.train(inputs, targets, eta=0.5, t=100)

xorerr_bp, xorerr_srn = xorerr(targets, inputs)

print("XOR Error:")
print("Backprop %f " % xorerr_bp)
print("SRN %f " % xorerr_srn)