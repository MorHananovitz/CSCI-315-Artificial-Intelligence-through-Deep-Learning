import numpy as np
import pickle

class Backprop:
    def __init__(self,n, m, h, part):
        self.input = n
        self.output = m
        self.hiddenunits = h
        self.WIH = np.random.random((n+1,h))-0.5
        self.WHO = np.random.random((h+1,m))-0.5
        self.part = part

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
        if self.part==1 :
            return O

        if self.part==2:
            return (O > 0.5)
        if self.part==3:
            max_loc = np.argmax(O)
            onehotvec=np.zeros(10)
            onehotvec[max_loc]=1
            return onehotvec
        if self.part == 4:
            return O, H

    def load_mine(self, name_file):
        return pickle.load(open(name_file, "rb"))

    def train(self,I, T, eta=0.5, mew = 0, t=10000, n_batch = 128):
        def save(Weights,name):
            pickle.dump(Weights, open(name, "wb"))

        def sigmoid(x, derivative=False):
            sigm = 1 / (1 + np.exp(-x))
            if derivative:
                return sigm * (1 - sigm)
            return sigm

        def batch_create(I, n_batch):
            return [I[i:i + n_batch,:] for i in range(0, len(I), n_batch)]

        batches_I,batches_T = batch_create(I, n_batch),batch_create(T, n_batch)

        RMSerr = []
        WIH_00 = []
        WHO_00 = []
        H_values=[]
        for i in range(t):
            if i %500 == 0:
                print("Complete %d / %d Iterations" % (i, t))

            dWIH = np.zeros((self.input + 1, self.hiddenunits))
            dWHO = np.zeros((self.hiddenunits + 1, self.output))
            dWIH_prev = 0
            dWHO_prev = 0

            for I_batch,T_batch in zip(batches_I,batches_T):

                Hnet = np.dot(np.hstack((I_batch,np.ones((I_batch.shape[0],1)))), self.WIH)
                H = sigmoid(Hnet)  #Sigmoid Function
                Onet = np.dot(np.hstack((H,np.ones((H.shape[0],1)))), self.WHO)
                O = sigmoid(Onet)
                delO = np.multiply((T_batch - O), sigmoid(O, True))
                err =  np.delete(np.dot(delO, self.WHO.T), -1, axis = 1)
                delH = np.multiply(err, sigmoid(Hnet, True)) #Backprop
                dWIH = dWIH + np.dot(np.hstack((I_batch,np.ones((I_batch.shape[0],1)))).T, delH)
                dWHO = dWHO + np.dot(np.hstack((H,np.ones((H.shape[0],1)))).T, delO)

                self.WIH = self.WIH + (eta*dWIH + mew*dWIH_prev)/len(I_batch)
                self.WHO = self.WHO + (eta*dWHO+ mew*dWHO_prev)/len(I_batch)

                dWIH_prev = dWIH
                dWHO_prev = dWHO

            if i%100==0:
                RMSerr.append(np.sum((T_batch - O)**2)/(len(I_batch)*self.output))
                WIH_00.append(self.WIH[1][0])
               # print(WIH_00)
                WHO_00.append(self.WHO[1][0])
                #H_values.

        save(self.WHO,"Part_%d_WHO_eta=%.1f_t=%d_h=%d.wgt" %(self.part, eta,t,self.hiddenunits))
        save(self.WIH,"Part_%d_WIH_eta=%.1f_t=%d_h=%d.wgt" %(self.part, eta,t,self.hiddenunits))

        print()
        print("Complete %d Iterations" % t)
        print()
        return eta, t, self.hiddenunits, mew, RMSerr, WIH_00, WHO_00