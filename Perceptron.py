#!/usr/bin/env python
# coding: utf-8

"""

@author: walidelmouahidi

"""

# %% Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% Perceptron


class Perceptron:
    """
    Perceptron is a simple binary classification algorithm.

    Attributes:
        r (float): Learning rate.
        X (numpy.ndarray): Input data matrix.
        y (numpy.ndarray): Target labels.
        w (numpy.ndarray): Weight vector.
        epoches (int): Number of training epochs.
        y_tmp (numpy.ndarray): Temporary predicted labels during training.
        v_tmp (float): Temporary weighted sum during training.
        x (numpy.ndarray): Input vector for a single data point.
        conv (int): Convergence flag (0 when converged, nonzero otherwise).

    Methods:
        activation(v): Activation function that returns -1 or 1 based on the input v.
        weights(): Train the perceptron to adjust its weights until convergence.
        prediction(inx): Predict the label for a new input.

    """
    
    def __init__(self, r, X, y):
        """
       Initialize the Perceptron with learning rate, input data, and target labels.

       Args:
           r (float): Learning rate.
           X (numpy.ndarray): Input data matrix.
           y (numpy.ndarray): Target labels.
       """
        self.y = y
        self.X = X
        self.r = r
        self.epoches = 0
        # Set the initial weights as a vector of dimension equal to a datapoint x
        self.w = np.zeros(self.X.shape[1]+1)
        self.y_tmp = np.zeros(len(self.X))
        self.v_tmp = 0
        self.x = np.ones(X.shape[1]+1)
        self.conv = 1
        self.weights()

    def activation(self, v):
        """
        Activation function that returns -1 for v <= 0 and 1 for v > 0.

        Args:
            v (float): Weighted sum of inputs.

        Returns:
            float: -1 or 1 based on the input v.
        """
        if v <= 0:
            tmp = -1.0
        else:
            tmp = 1.0
        return tmp

    def weights(self,):
        """
        Train the perceptron to adjust its weights until convergence.

        Returns:
            tuple: A tuple containing the learned weights and the number of training epochs.
        """
        while self.conv != 0:
            self.conv = sum(self.y-self.y_tmp)
            self.epoches = self.epoches+1
            for i in range(len(self.X)):
                self.x[1:] = self.X[i, :]
                self.v_tmp = np.dot(self.w, self.x)
                self.y_tmp[i] = self.activation(self.v_tmp)
                self.w = self.w+(self.r/2)*(self.y[i]-self.y_tmp[i])*self.x
        return self.w, self.epoches

    def prediction(self, inx):
        """
        Predict the label for a new input.

        Args:
            inx (numpy.ndarray): Input data point.

        Returns:
            float: Predicted label (-1 or 1).
        """
        tmp_inx = np.ones(inx.shape[0]+1)
        tmp_inx[1:] = inx
        self.outy = self.activation(np.dot(self.w, tmp_inx))
        return self.outy


# %% Test 1


set_x = np.array([[4, 3], [3, 11], [-3, -2],[0,4],[-2,8],[8,6.5],[10,10],[2,-4],[4,-4.5],
                 [5, -1], [9, 1], [8, -3], [10, -5]])
set_y = np.array([1, 1, 1,1,1,1,1,-1,-1, -1, -1, -1, -1])

w_test, e_test = Perceptron(0.5, set_x, set_y).weights()
yy = Perceptron(0.5, set_x, set_y).prediction(np.array([0, 10]))
a = Perceptron(0.5, set_x, set_y).activation(0)
print(a)
print(yy)
print(w_test, e_test)


# %% Test 2


print(w_test, e_test)
plt.plot(set_x[:7, 0], set_x[:7, 1], "ro")
plt.plot(set_x[7:, 0], set_x[7:, 1], "bo")
x = np.linspace(-5, 10, 100)
y = -(w_test[1]/w_test[2])*x-w_test[0]/w_test[2]
plt.plot(x, y, "k-")
plt.show
