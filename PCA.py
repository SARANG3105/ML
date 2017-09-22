# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 18:44:57 2017

@author: user
"""
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
#importing data from the dataset
Data=np.loadtxt('C:\Users\user\Desktop\R folder\dataset_1.csv',skiprows=1, delimiter=',')
#separate vales of x, y and z
def PCA(Data):
    x=Data[:,0]
    y=Data[:,1]
    z=Data[:,2]
#new values after subtracting mean
    new_x= x- np.mean(x)
    new_y= y- np.mean(y)
    new_z= z- np.mean(z)
#new dataset after subtracting mean from each column
    new_data= np.column_stack((new_x,new_y,new_z))
    data_trans=new_data.T
#covariance matrix of new_data
    Cov_mat= np.dot(data_trans,new_data)/len(x-1)
#EigenValue and EigenVector of Covariance Matrix
    eiganValue,eiganVector= LA.eig(Cov_mat)
#sorted eigenValues
    np.sort(eiganValue)
#selected columns of eigen vector with largest eigen values.
    maxEiganVal=np.argmax(eiganValue)

    newVector= eiganVector[:,maxEiganVal]
#formed Principal Components
    pc= np.dot(new_data,newVector)
#
    #pc1= pc[0,:]
    #pc2= pc[1,:]

    #fig= plt.figure()
    #Ax= fig.add_subplot(1, 1, 1)
    #Ax.plot(pc1,pc2)
    #fig.show()

PCA(Data)

