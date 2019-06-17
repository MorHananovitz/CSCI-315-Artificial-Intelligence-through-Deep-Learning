import numpy as np

class Backprop:
    def __init__(self,n, m, h):
        self.input = n
        self.output = m
        self.hiddenunits = h
        self.WIH = np.random.random((n+1,h))-0.5
        self.WHO = np.random.random((h+1,m))-0.5

    def __str__(self):
        return str( "This is a Perceptron with %d inputs and %d outputs and %d hidden units" % (self.input, self.output, self.hiddenunits))

    def test(self, I):
        def sigmoid(x, derivative=False):
            sigm = 1 / (1 + np.exp(-x))
            if derivative:
                return sigm * (1 - sigm)
            return sigm

        Iadjust = np.hstack((np.array(I), np.ones(len(I)).reshape(-1,1)))
        Hnet = np.dot(Iadjust, self.WIH)
        H = sigmoid(Hnet)  # Sigmoid Function
        Onet = np.dot(np.hstack((H, np.ones(len(H)).reshape(-1,1))), self.WHO)
        O = sigmoid(Onet)
        return O

    def train(self,I, T, eta=0.5, mew = 0, t=10000):
        def sigmoid(x, derivative=False):
            sigm = 1 / (1 + np.exp(-x))
            if derivative:
                return sigm * (1 - sigm)
            return sigm

        RMSerr = np.zeros(t)
        I = np.arange(len(I)).reshape(len(I),1)
        p = len(I)

        for i in range(t):
            if i % 10 == 0:
                print("Complete %d / %d Iterations" % (i, t))
            dWIH = np.zeros((self.input + 1, self.hiddenunits))
            dWHO = np.zeros((self.hiddenunits + 1, self.output))

            error = 0

            for j in range(len(I)):
                Hnet = np.dot(np.append(I[j], 1), self.WIH)
                H = sigmoid(Hnet)  # Sigmoid Function
                Onet = np.dot(np.append(H, 1), self.WHO)
                O = sigmoid(Onet)
                delO = np.multiply((T[j] - O), sigmoid(O, True))
                err = np.dot(delO, self.WHO.T)[:-1]
                delH = np.multiply(err, sigmoid(Hnet, True))  # Backprop
                dWIH = dWIH + np.dot(np.append(I[j], 1).reshape(-1, 1), delH.reshape(-1, 1).T)
                dWHO = dWHO + np.dot(np.append(H, 1).reshape(-1, 1), delO.reshape(-1, 1).T)

                self.WIH = self.WIH + eta * dWIH
                self.WHO = self.WHO + eta * dWHO
                error += np.power((T[i] - O), 2)

            RMSerr[i] = np.sqrt(np.sum(error)/p)

            if (i + 1) % 10 == 0 and (i + 1 < t):
                print(RMSerr[i])

            #if i%10==0:
             #   RMSerr.append(np.sum((T - O) ** 2) / (len(I)))
              #  print(np.sum((T - O) ** 2) / (len(I)))

        print()
        print("Complete %d Iterations for Backprop" % t)
        return eta, t, self.hiddenunits, RMSerr