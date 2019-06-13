import numpy as np

class SRN:
    def __init__(self,n, m, h):
        self.input = n
        self.output = m
        self.hiddenunits = h
        self.WIH = np.random.random((n+h+1,h))-0.5
        self.WHO = np.random.random((h+1,m))-0.5

    def __str__(self):
        return str( "This is a Perceptron with %d inputs and %d outputs and %d hidden units" % (self.input, self.output, self.hiddenunits))

    def test(self, I):

        def sigmoid(x, derivative=False):
            sigm = 1 / (1 + np.exp(-x))
            if derivative:
                return sigm * (1 - sigm)
            return sigm

        aH = np.zeros(2)
        for row_I in I:
            contex_layer = np.append(aH, np.append(row_I, 1))
            Hnet = np.dot(contex_layer, self.WIH)
            H = sigmoid(Hnet)  # Sigmoid Function
            aH = H
            Onet = np.dot(np.append(H, 1), self.WHO)
            O = sigmoid(Onet)
        return O

    def train(self,I, T, eta=0.5, mew = 0, t=10000):
        def sigmoid(x, derivative=False):
            sigm = 1 / (1 + np.exp(-x))
            if derivative:
                return sigm * (1 - sigm)
            return sigm

        RMSerr = []
        for i in range(t):

            if i % 10 == 0:
                print("Complete %d / %d Iterations" % (i, t))
            dWIH = np.zeros((self.input + self.hiddenunits + 1, self.hiddenunits))
            dWHO = np.zeros((self.hiddenunits + 1, self.output))

            aH = np.zeros(2)
            for row_I, row_T in zip(I, T):
                contex_layer = np.append(aH, np.append(row_I, 1))
                Hnet = np.dot(contex_layer, self.WIH)
                H = sigmoid(Hnet)  # Sigmoid Function
                aH= H
                Onet = np.dot(np.append(H, 1), self.WHO)
                O = sigmoid(Onet)
                delO = np.multiply((row_T - O), sigmoid(O, True))
                err = np.dot(delO, self.WHO.T)[:-1]
                delH = np.multiply(err, sigmoid(Hnet, True))  # Backprop
                dWIH = dWIH + np.dot(contex_layer.reshape(-1, 1), delH.reshape(-1, 1).T)
                dWHO = dWHO + np.dot(np.append(H, 1).reshape(-1, 1), delO.reshape(-1, 1).T)

                self.WIH = self.WIH + eta * dWIH
                self.WHO = self.WHO + eta * dWHO

            if i%10==0:
                RMSerr.append(np.sum((T - O) ** 2) / 3000)
                print(np.sum((T - O) ** 2) / 3000)


        print()
        print("Complete %d Iterations for SRN" % t)
        print()
        return eta, t, self.hiddenunits, RMSerr