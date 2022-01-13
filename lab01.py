# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:59:04 2022

@author: Sreelaksmi
"""

import numpy as np

#input data
x_all = np.array(([2,9],[1,5],[3,6],[5,10],[10,6]),dtype=float)

#Set teh target variable
y= np.array(([92],[86],[89],[70],[86]),dtype=float)



#Normalisation - Scaling the input data
x_all = x_all/np.max(x_all,axis=0)

#Normalisation - Scaling the output data
y=y/np.max(y)

#Splitting the data into test data and train data
X = np.split(x_all,[3])[0]
X_test = np.split(x_all,[3])[1]

Y=np.split(y,[3])[0]
Y_test=np.split(y,[3])[1]

#class defining neural network with the parameter such as input, hidden and output layer


#Forward Propagation
class neural_network(object):
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        
        #initializing the weights with the random values - input to hidden layer
        self.W1 = np.random.randn(self.inputSize,self.hiddenSize)
        
        #initializing the weights to hidden to output layer
        self.W2 = np.random.randn(self.hiddenSize,self.outputSize)
        
    def sigmoid(self, s):
        return 1/(1+np.exp(-s))
    
    #Now onto the forward funtion
    def forward(self,X):
        #dot product of X and first set of weights
        self.z = np.dot(X,self.W1)
        
        #Activation function
        self.z2 = self.sigmoid(self.z)
        
        #dot product of z2 and second set of weights
        self.z3 = np.dot(self.z2,self.W2)
        
        #Activation function
        o = self.sigmoid(self.z3)
        return o
    #function for defing the derivative of the sigmoid function
    def sigmoidPrime(self,s):
        return s*(1-s)
    
    #implementing the backward propagation
    def backPropagation(self,X,y,o):
        #finding the error at the output
        self.o_error = y-o
        #Applying the derivative of sigmoid to error
        self.o_delta = self.o_error*self.sigmoidPrime(o)
        
        #finding how much the hidden layer contributes the error to the output layer
        self.o_error_hidden = self.o_delta.dot(self.W2.T)
        
        #applying the derivative to o_error_hidden
        self.o_delta_hidden = self.o_error_hidden*self.sigmoidPrime(self.z2)
        
        #Adjusting the W1 and W2
        self.W1 += X.T.dot(self.o_delta_hidden)
        self.W2 += self.z2.T.dot(self.o_delta)
    def train(self,X,y):
        o=self.forward(X)
        self.backPropagation(X, y, o)
    def saveWeights(self):
        np.savetxt("w1.txt",self.W1,fmt="%s")
        np.savetxt("w2.txt",self.W2,fmt="%s")
    def predict(self):
        print("Predicted data based on the trained weights:")
        print("Input (Scaled):\n"+str(X_test))
        print("Predicted output:\n"+str(self.forward(X_test)))
        print("Actual Output: \n"+str(Y_test))
        
        
    
#defining neural network
nn = neural_network()

for i in range(1000):
    print("Input: \n"+str(X))
    print("Actual Output: \n"+str(Y))
    print("Predicted Output: \n"+str(nn.forward(X)))
    #training
    nn.train(X, Y)
    print("Loss:\n"+str(np.mean(np.square(Y-nn.forward(X)))))
    
nn.saveWeights()
nn.predict()
    
        
        
        
        




