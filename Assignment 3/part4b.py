import numpy as np
import itertools as it
import pandas as pd
from backpropfast import Backprop
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
eta, t, h, mew, RMSerr, WIH_00, WHO_00 = NN.train(np.array(input_train), np.array(targets), eta=0.05, t=1000, n_batch = 1000)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x = np.arange(0, t, 100)
y = np.array(RMSerr)
ax.plot(x, y)
ax.set_xlabel('Iterations')
ax.set_ylabel('RMS error')
plt.title("Part 4b Plot")
plt.show()