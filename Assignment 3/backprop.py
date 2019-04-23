import numpy as np
import pickle

class Backprop:
    def __init__(self,n, m, h):
        self.input = n
        self.output = m
        self.hiddenunits = h
        self.WIH = np.random.random((n+1,h))-0.5
        self.WHO = np.random.random((h+1,m))-0.5

    def __str__(self):
        return str( "This is a Perceptron with %d inputs and %d outputs and %d hidden units" % (self.input, self.output, self.hiddenunits))

    def test(self, J, T):
        def sigmoid(x, derivative=False):
            sigm = 1 / (1 + np.exp(-x))
            if derivative:
                return sigm * (1 - sigm)
            return sigm

        Hnet = np.dot(np.append(J, 1), self.WIH)
        H = sigmoid(Hnet)  # Sigmoid Function
        Onet = np.dot(np.append(H, 1), self.WHO)
        O = sigmoid(Onet)
        return(np.abs(O-T))

    def save(self):
        pickle.dump(self.WIH, open("WIH_prev.p", "w"))

    def load(self):
        pickle.load(open("WIH_prev.p", "r"))

    def train(self,I, T, eta=0.5, t=1000):
        def sigmoid(x, derivative=False):
            sigm = 1 / (1 + np.exp(-x))
            if derivative:
                return sigm * (1 - sigm)
            return sigm

        for i in range(t):
            if i %100 == 0:
                print("Complete %d / %d Iterations" % (i, t))
            dWIH = np.zeros((self.input + 1, self.hiddenunits))
            dWHO = np.zeros((self.hiddenunits + 1, self.output))

            for row_I, row_T in zip(I, T):
                Hnet = np.dot(np.append(row_I, 1), self.WIH)
                H = sigmoid(Hnet)  #Sigmoid Function
                Onet = np.dot(np.append(H, 1), self.WHO)
                O = sigmoid(Onet)
                delO = np.multiply((row_T - O), sigmoid(O, True))
                err =  np.dot(delO, self.WHO.T)[:-1]
                delH = np.multiply(err, sigmoid(Hnet, True)) #Backprop
                dWIH = dWIH + np.dot(np.append(row_I, 1).reshape(-1,1), delH.reshape(-1,1).T)
                dWHO = dWHO + np.dot(np.append(H, 1).reshape(-1,1), delO.reshape(-1,1).T)

            self.WIH = self.WIH + eta*dWIH / len(I)
            self.WHO = self.WHO + eta*dWHO/ len(I)

        print()
        print("Complete %d Iterations" % t)