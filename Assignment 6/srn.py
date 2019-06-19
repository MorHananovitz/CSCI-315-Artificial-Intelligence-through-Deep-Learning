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

        O = []
        aH = np.zeros(self.hiddenunits)
        for row_I in I:
            contex_layer = np.append(aH, np.append(row_I, 1))
            Hnet = np.dot(contex_layer, self.WIH)
            H = sigmoid(Hnet)  # Sigmoid Function
            aH = H
            Onet = np.dot(np.append(H, 1), self.WHO)
            O_help = sigmoid(Onet)
            O.append(O_help)
        return O

    def train(self,I, T, eta=0.5, t=10000):
        def sigmoid(x, derivative=False):
            sigm = 1 / (1 + np.exp(-x))
            if derivative:
                return sigm * (1 - sigm)
            return sigm

        RMSerr = np.zeros(t)
        p = len(I)

        for i in range(t):

            aH = np.zeros(self.hiddenunits)
            error = 0

            for j in range(len(I)):
                contex_layer = np.append(aH, np.append(I[j], 1))
                Hnet = np.dot(contex_layer, self.WIH)
                H = sigmoid(Hnet)  # Sigmoid Function
                aH= H
                Onet = np.dot(np.append(H, 1), self.WHO)
                O = sigmoid(Onet)
                delO = np.multiply((T[j] - O), sigmoid(O, True))
                err = np.dot(delO, self.WHO.T)[:-1]
                delH = np.multiply(err, sigmoid(Hnet, True))  # Backprop
                dWIH = np.dot(contex_layer.reshape(-1, 1), delH.reshape(-1, 1).T)
                dWHO = np.dot(np.append(H, 1).reshape(-1, 1), delO.reshape(-1, 1).T)

                self.WIH += eta * dWIH
                self.WHO += eta * dWHO
                error += np.power((T[j] - O), 2)

            RMSerr[i] = np.sqrt(np.sum(error)/p)

            if (i)%10==0:
                print("%d / %d: %.6f" % (i, t, RMSerr[i]))

        print()
        print("Complete %d Iterations for SRN" % t)
        print()
        return eta, t, self.hiddenunits, RMSerr