import numpy as np
import pandas as pd
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