import numpy as np
import pandas as pd
import re

def Vector2num(target):
    return np.where(target > 0)[0]

data = pd.read_csv('digits_train.txt', header=None)

def dataread(data):
    for i in range(1):
        list_data=[]
        for col in data.values[16*i + 1: 16*i + 15]:
            list_data.append([int(x) for x in re.findall('\d+', col[0])])
        input_vector = np.concatenate(list_data)
        input_vector_visulize= np.array(input_vector > 0)*1
        A = input_vector_visulize.reshape((14, 14))
        for row in A:
            for j in row:
                if  j==0:
                    print(' ', end = '')
                else:
                    print('*', end = '')
            print()

        # Creating the target (str 2 num)
        target = np.array([int(x) for x in re.findall('\d+', data.values[16*i + 15][0])])
        # Convert vector to target
        t = Vector2num(target)
        #print(input_vector_visulize.reshape((14,14)))
        print('The target is: %d' % t)

dataread(data)