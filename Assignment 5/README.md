# Assignment 5

### Model Parameters:
- Learning rate: 0.0001
- Epochs: 8
- Batch Size: 100

### Architecture
- Number of hidden units 1 = 256
- Number of hidden units 1 = 256

### Results:
  ```sh
Epoch: 0008 cost: 0.023067260
Validation Error: 0.008199989795684814
Test Accuracy: **0.9913**
```
###Confusion Matrix:
  ```sh
       0     1     2     3    4    5    6     7    8    9
0  977     0     0     0    0    0    2     1    0    0
1    0  1133     2     0    0    0    0     0    0    0
2    1     1  1027     0    0    0    0     3    0    0
3    0     0     2  1002    0    2    0     2    2    0
4    0     0     0     0  978    0    1     0    0    3
5    2     0     0     4    0  884    1     0    0    1
6    3     2     0     1    2    6  944     0    0    0
7    0     3     5     1    0    0    0  1016    1    2
8    3     0     3     1    0    1    0     2  961    3
9    0     3     0     2    6    2    0     3    2  991
```