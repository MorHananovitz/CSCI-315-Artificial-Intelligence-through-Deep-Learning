import numpy as np

#Part 1 - Build Your Perceptron (Batch Learning)
class Perceptron:
    def __init__(self,n, m):
        self.input = n
        self.output = m
        self.weight = np.random.random((n+1,m))-0.5

    def __str__(self):
        return str( "This is a Perceptron with %d inputs and %d outputs" % (self.input, self.output))

    def test(self, J):
        return(np.dot(np.transpose(np.append(J, [1])), self.weight) > 0)

    def train(self,I,T, t=1000):
        Itrain = np.hstack((I, np.split(np.ones(len(I)), len(I))))
        for i in range(t):
            if i %100 == 0:
                print("Complete %d / %d Iterations" % (i, t))
            dW = np.zeros((self.input + 1, self.output))
            for row_I, row_T in zip(Itrain, T):
                O = np.dot(row_I, self.weight) > 0
                D = row_T - O
                dW = dW + np.outer(row_I, D)
            self.weight = self.weight + dW / len(Itrain)
        print()
        print("Complete %d Iterations" % t)

