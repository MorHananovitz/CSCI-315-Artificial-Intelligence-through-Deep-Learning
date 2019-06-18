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

def xorerr(targets, inputs, bp, srn):
    Obp = bp.test(inputs)
    Osrn = srn.test(inputs)

    xorerr_bp = 0.0
    xorerr_srn = 0.0

    p = (len(inputs)/3)

    for count in range(len(targets)):
        if count % 3 == 1:
            xorerr_bp += (Obp[count] - targets[count]) ** 2
            xorerr_srn += (Osrn[count] - targets[count]) ** 2
        RMS_bp = np.sqrt(xorerr_bp/p)
        RMS_srn = np.sqrt(xorerr_srn/p)

    return RMS_bp, RMS_srn #

inputs = xorseq(1000)
targets = shift(inputs)

print("Backprop Results")
bp = Backprop(1, 1, 20) #1 input, 1 outputs, 20 hidden units
eta, t, hiddenunits, RMSerr=bp.train(inputs, targets, eta=0.5, t=100)

print()
print("SRN Results")
srn = SRN(1, 1, 20) #1 input, 1 outputs, 20 hidden units
srn.train(inputs, targets, eta=0.5, t=100)

RMS_bp, RMS_srn = xorerr(targets, inputs, bp, srn)

print("XOR Error:")

print("Backprop: %.6f " % RMS_bp)
print("SRN: %.6f " % RMS_srn)