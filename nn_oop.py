import numpy as np


'''
Hare we are going to implement neural network from scratch in python using OOP.

Things we are going to try
        1} Different Optimization technique(Adam,gradient Descent,gradient descent with L2 reg)
        2} we will see how to implement Dropout for Regularization
        3} Finally Implementation of different activation function

'''
class Nn(object):

    '''
    hare we are initializing the NN with number of hidden units in layer 1 and layer  2 and layer 3
    also with number of outputs for classification and Regression type
    num_outputs should be 1 in case of Regression Task
    '''

    def __init__(self,input_dim,num_hidden_units_L1,num_hidden_units_L2,output_dim,num_outputs):
        # 3-layer neural network
        self.input_dim = input_dim
        self.num_hidden_units_L1 = num_hidden_units_L1
        self.num_hidden_units_L2 = num_hidden_units_L2
        self.output_dim = output_dim
        self.num_outputs = num_outputs

        # initializing the hyper parameters for neural network
    def init_params(self):
        # weight
        self.w1 = 2*np.random.randn(self.num_hidden_units_L1,self.input_dim)-1
        self.w2 = 2*np.random.randn(self.num_hidden_units_L2,self.num_hidden_units_L1)-1
        self.w3 = 2*np.random.randn(self.output_dim,self.num_hidden_units_L2)-1

        # biases
        self.b1 = np.zeros(shape = (self.num_hidden_units_L1,1))
        self.b2 = np.zeros(shape = (self.num_hidden_units_L2,1))
        self.b3 = np.zeros(shape = (1,self.output_dim))


        #print("The Weights and Biases are Initialized")

        self.param_grid = {'w1':self.w1,'w2':self.w2,'w3':self.w3,
                           'b1':self.b1,'b2':self.b2,'b3':self.b3}
        return self.param_grid



    '''
        hare we are going with the forward propagation and Dropout implemented.
        general equation for forward_prop is 
        ------>out_put = f(weights*input + biases)
               where f is an activation function
               
               Different Activation Function avialable are 
               ReLU(Rectified Linear Unit)
               Sigmoid(Used For Binary Classification)
               tanh
    '''
    def forward_prop(self,input_data=None):

        # Time for forward propagation
        self.z1 = np.dot(self.w1,input_data.T) + self.b1
        self.a1 = self.ReLU(self.z1)

        # Implementing dropout on 1st layer
        self.a1 = self.Dropout(drop_out = 0.5,hidden_layer = self.a1,shape_0 = self.a1.shape[0],shape_1 = self.a1.shape[1])
        self.z2 = np.dot(self.w2,self.a1) + self.b2
        self.a2 = self.ReLU(self.z2)

        # Implementing dropout in 2nd layer
        self.a2 = self.Dropout(drop_out=0.5, hidden_layer=self.a2,shape_0 =self.a2.shape[0],shape_1 = self.a2.shape[1])
        self.z3 = np.dot(self.w3,self.a2) + self.b3
        if self.num_outputs == 2:
            self.a3 = self.sigmoid(self.z3)
        elif self.num_outputs > 2:
            self.a3 = self.softmax(self.z3)
        else:
            self.a3 = self.ReLU(self.z3)

        self.cache = {'a1':self.a1,'a2':self.a2,'a3':self.a3,'z1':self.z1,'z2':self.z2,'z3':self.z3}

        return self.a3

    """
    Implementation of drop out
    background:---
        It randomly knocks out some of the neurons based on the drop_out parameter used.
        Simply it is a type of regularization technique which simply makes some of the neurons inactive randomly.
    """

    def Dropout(self,drop_out,hidden_layer,shape_0,shape_1):

        '''
            If no drop out is required then use drop_out = 1
            else give the value in 0 to 1 e.g 0.7
        '''

        self.d = np.random.rand(shape_0,shape_1) < drop_out
        self.out = np.multiply(hidden_layer,self.d)
        self.out = self.out/drop_out
        return self.out

    """
    Implementation of Back propagation.    
    See ReadMe
    """

    def back_prop(self,input_data,y,learning_rate):

        m = y.shape[0]
        self.dz3 = self.a3 - y.T
        self.dw3 = 1/m*np.dot(self.dz3,self.a2.T)
        self.db3 = 1/m*np.sum(self.dz3,axis = 1,keepdims=True)

        self.dz2 = np.multiply(np.dot(self.w3.T, self.dz3),self.ReLU_derivative(self.a2))
        self.dw2 = 1 / m * np.dot(self.dz2, self.a1.T)
        self.db2 = 1 / m * np.sum(self.dz2, axis=1, keepdims=True)

        self.dz1 = np.multiply(np.dot(self.w2.T, self.dz2),self.ReLU_derivative(self.a1))
        self.dw1 = 1 / m * np.dot(self.dz1, input_data)
        self.db1 = 1 / m * np.sum(self.dz1, axis=1, keepdims=True)

        self.w1 -= learning_rate*self.dw1
        self.w2 -= learning_rate*self.dw2
        self.w3 -= learning_rate*self.dw3

        self.b1 -= learning_rate*self.db1
        self.b2 -= learning_rate*self.db2
        self.b3 -= learning_rate*self.db3

        self.param_grid = {'w1':self.w1,'w2':self.w2,'w3':self.w3,
                           'b1':self.b1,'b2':self.b2,'b3':self.b3}
        return self.param_grid

    # Back Prop with L2 Regularizer
    def back_prop_with_L2_reg(self,input_data,y,learning_rate,lmda):

        m = y.shape[0]
        self.dz3 = self.a3 - y.T
        self.dw3 = 1 / m * np.dot(self.dz3, self.a2.T)+lmda/m*self.w3
        self.db3 = 1 / m * np.sum(self.dz3, axis=1, keepdims=True)

        self.dz2 = np.multiply(np.dot(self.w3.T, self.dz3), self.ReLU_derivative(self.a2))
        self.dw2 = 1 / m * np.dot(self.dz2, self.a1.T)+lmda/m*self.w2
        self.db2 = 1 / m * np.sum(self.dz2, axis=1, keepdims=True)

        self.dz1 = np.multiply(np.dot(self.w2.T, self.dz2), self.ReLU_derivative(self.a1))
        self.dw1 = 1 / m * np.dot(self.dz1, input_data)+lmda/m*self.w1
        self.db1 = 1 / m * np.sum(self.dz1, axis=1, keepdims=True)

        self.w1 = self.w1*(1-learning_rate*lmda)/m-learning_rate*self.dw1
        self.w2 = self.w2*(1-learning_rate*lmda)/m-learning_rate*self.dw2
        self.w3 = self.w3*(1-learning_rate*lmda)/m-learning_rate*self.dw3

        self.b1 -= learning_rate * self.db1
        self.b2 -= learning_rate * self.db2
        self.b3 -= learning_rate * self.db3

        self.param_grid = {'w1': self.w1, 'w2': self.w2, 'w3': self.w3,
                           'b1': self.b1, 'b2': self.b2, 'b3': self.b3}
        return self.param_grid


    # Adam optimizer
    def adam_optimizer(self,input_data,y,learning_rate):

        beta1 = 0.9
        beta2 = 0.999
        epsilon = 10^-8
        m = y.shape[0]
        self.dz3 = self.a3 - y.T
        self.dw3 = 1 / m * np.dot(self.dz3, self.a2.T)
        self.db3 = 1 / m * np.sum(self.dz3, axis=1, keepdims=True)

        self.dz2 = np.multiply(np.dot(self.w3.T, self.dz3), self.ReLU_derivative(self.a2))
        self.dw2 = 1 / m * np.dot(self.dz2, self.a1.T)
        self.db2 = 1 / m * np.sum(self.dz2, axis=1, keepdims=True)

        self.dz1 = np.multiply(np.dot(self.w2.T, self.dz2), self.ReLU_derivative(self.a1))
        self.dw1 = 1 / m * np.dot(self.dz1, input_data)
        self.db1 = 1 / m * np.sum(self.dz1, axis=1, keepdims=True)

        ##after runing with back prop we are going to implement the credentials of Adam

        # initializing the base parameters
        #vdw1,vdw2,vdw3,vdb1,vdb2,vdb3,sdw1,sdw2,sdw3,sdb1,sdb2,sdb3 = 0,0,0,0,0,0,0,0,0,0,0,0


        ##This rule for momentum part
        vdw1 = beta1 * self.dw1 + (1 - beta1) * self.dw1
        vdw2 = beta1 * self.dw2 + (1 - beta1) * self.dw2
        vdw3 = beta1 * self.dw3 + (1 - beta1) * self.dw3

        vdb1 = beta1 * self.db1 + (1 - beta1) * self.db1
        vdb2 = beta1 * self.db2 + (1 - beta1) * self.db2
        vdb3 = beta1 * self.db3 + (1 - beta1) * self.db3

        ##This rule is for RMSPRop part
        sdw1 = beta1 * self.dw1 + (1 - beta2) * np.square(self.dw1)
        sdw2 = beta1 * self.dw2 + (1 - beta2) * np.square(self.dw2)
        sdw3 = beta1 * self.dw3 + (1 - beta2) * np.square(self.dw3)

        sdb1 = beta1 * self.db1 + (1 - beta2) * np.square(self.db1)
        sdb2 = beta1 * self.db2 + (1 - beta2) * np.square(self.db2)
        sdb3 = beta1 * self.db3 + (1 - beta2) * np.square(self.db3)


        ## Time for update rule for the parameters
        ## We havent implemented Bias correction

        self.w1 -= learning_rate * vdw1 #/ np.sqrt((sdw1+epsilon))
        self.w2 -= learning_rate * vdw2 #/ np.sqrt((sdw2+epsilon))
        self.w3 -= learning_rate * vdw3 #/ np.sqrt((sdw3+epsilon))

        self.b1 -= learning_rate * vdb1 #/ np.sqrt((sdb1+epsilon))
        self.b2 -= learning_rate * vdb2 #/ np.sqrt((sdb2+epsilon))
        self.b3 -= learning_rate * vdb3 #/ np.sqrt((epsilon+sdb3))

        self.param_grid = {'w1': self.w1, 'w2': self.w2, 'w3': self.w3,
                           'b1': self.b1, 'b2': self.b2, 'b3': self.b3}
        return self.param_grid


    def data_normalization(self,X):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X = sc.fit_transform(X)
        return X


    def train(self,X,y):
        cache = self.forward_prop(input_data = X)
        model = self.back_prop_with_L2_reg(input_data=X,y=y,learning_rate=0.007,lmda = 4)
        return model

    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    def ReLU(self,x):
        return np.maximum(x,0)

    def ReLU_derivative(self,x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def sigmoid_derivative(self,x):
        return x * (1 - x)

    def tanh(self,x):
        return np.tanh(x)

    def tanh_derivative(self,x):
        return (1-np.power(x,2))

    def log_loss(self,y, y_hat):
        try:
            N = y.shape[0]
            l = -1 / N * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
            return l
        except ZeroDivisionError:
            pass


    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

from sklearn.datasets import load_breast_cancer
can = load_breast_cancer()
X = can.data
y = can.target

nn_1 = Nn(30,60,40,1,2)
losses = []
#nn_1.data_normalization(X)
nn_1.init_params()
for i in range(5000):
    out = nn_1.forward_prop(X)
    losses.append(nn_1.log_loss(y.T,out))
    nn_1.train(X,y)

import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()