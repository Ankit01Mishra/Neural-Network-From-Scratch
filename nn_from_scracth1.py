import numpy as np
#Seed the random function to ensure that we always get the same result
np.random.seed(1)
#Variable definition
#set up w0
W1 = 2*np.random.random((60,30)) - 1
W2 = 2*np.random.random((30,60)) - 1
W3 = 2*np.random.random((1,30))-1

#define X
from sklearn.datasets import load_breast_cancer
can = load_breast_cancer()
X = can.data
y = can.target

#b may be 0
b1 = np.zeros(shape = (60,1))
b2 = np.zeros(shape = (30,1))
b3 = np.zeros(shape = (1,1))
m = X.shape[0]
A0 = X.T

# sigmoid function
def sigmoid(x):
    try:

        return 1/(1+np.exp(-x))
    except RuntimeError:
        pass

def sigmoid_derivative(x):
    return x*(1-x)


#Log Loss function
def log_loss(y,y_hat):
    N = y.shape[0]
    l = -1/N * np.sum(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
    return l

def log_loss_derivative(y,y_hat):
    return (y_hat-y)


losses = []
for i in range(4000):
    # do the linear step
    z1 = np.dot(W1,A0)+b1
    # pass the linear step through the activation function
    A1 = sigmoid(z1)
    #print(A1.shape)

    z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(z2)

    z3 = np.dot(W3,A2)+b3
    A3 = sigmoid(z3)
    #print(A2.shape)

    # Calculate loss
    loss = log_loss(y=y, y_hat=A3)

    # we use this to keep track of losses over time
    losses.append(loss)
    #print(y.shape)
    # Calculate derivative of L(w) with respect to z0
    dz3 = A3-y.T
    #print(dz2.shape)

    # Calculate derivative of L(w) with respect to w0
    dW3 = 1 / m * np.dot(dz3,A2.T)

    # Calculate derivative of L(w) with respect to b0
    db3 = 1 / m * np.sum(dz3, axis=1, keepdims=True)
    #print(db2.shape)

    dz2 = np.multiply(np.dot(W3.T,dz3),sigmoid_derivative(A2))
    dW2 = 1/m*np.dot(dz2,A1.T)
    db2 = 1/m*np.sum(dz2,axis = 1,keepdims=True)

    dz1 = np.multiply(np.dot(W2.T, dz2), sigmoid_derivative(A1))
    dW1 = 1 / m * np.dot(dz1, A0.T)
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)
    #print(db1.shape)

    # Update w0 accoarding to gradient descent algorithm
    # To keep things easy we will ignore the learning rate alpha for now, it will be covered in the next chapter
    W1 -= 0.009*dW1
    W2 -= 0.009*dW2
    W3 -= 0.009*dW3

    # Update b0 accoarding to gradient descent algorithm
    b1 -= 0.009*db1
    b2 -= 0.009*db2
    b3 -= 0.009*db3

#Plot losses over time
#As you can see our algorithm does quite well and quickly reduces losses
import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()

