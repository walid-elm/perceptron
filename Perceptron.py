#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[135]:


## Perceptron

class Perceptron:
    
    def __init__(self,r,X,y):
        self.y=y
        self.X=X
        self.r=r
        self.epoches=0
        self.w=np.zeros(self.X.shape[1]+1) #set the initial weights as a vector of dimension equal to a datapoint x
        self.y_tmp=np.zeros(len(self.X))
        self.v_tmp=0
        self.x=np.ones(X.shape[1]+1)
        self.conv=1
        self.weights()
        
        
    def activation(self,v):
        if v<=0:
            tmp=-1.0
        else:
            tmp=1.0
        return tmp
    
    def weights(self,):
        while self.conv!=0:
            self.conv=sum(self.y-self.y_tmp)
            self.epoches=self.epoches+1
            for i in range(len(self.X)):
                self.x[1:]=self.X[i,:]
                self.v_tmp=np.dot(self.w,self.x)
                self.y_tmp[i]=self.activation(self.v_tmp)
                self.w=self.w+(self.r/2)*(self.y[i]-self.y_tmp[i])*self.x
        return self.w, self.epoches
    
    def prediction(self,inx):
        tmp_inx=np.ones(inx.shape[0]+1)
        tmp_inx[1:]=inx
        self.outy=self.activation(np.dot(self.w,tmp_inx))
        return self.outy
    


# In[136]:


set_x=np.array([[4,3],[3,11],[-3,-2],[5,-1],[9,1],[8,-3],[10,-5]])
set_y=np.array([1,1,1,-1,-1,-1,-1])

w_test, e_test=Perceptron(0.5,set_x,set_y).weights()
yy=Perceptron(0.5,set_x,set_y).prediction(np.array([0,10]))
a=Perceptron(0.5,set_x,set_y).activation(0)
print(a)
print(yy)
print(w_test, e_test)


# In[137]:


print(w_test, e_test)
plt.plot(set_x[:3,0],set_x[:3,1], "ro")
plt.plot(set_x[3:,0],set_x[3:,1], "bo")
x=np.linspace(-5,10,100)
y=-(w_test[1]/w_test[2])*x-w_test[0]/w_test[2]
plt.plot(x,y,"k-")
plt.show


# In[126]:


np.ones([1,2])


# In[132]:


np.ones(2).shape[0]


# In[ ]:




