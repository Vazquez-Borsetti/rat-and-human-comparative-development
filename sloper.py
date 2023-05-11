# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 09:32:21 2021

@author: deses
"""
import numpy as np
import matplotlib.pyplot as plt
from quantileregresion5 import QLRegression2

def slopes(data,quantile,a,b):
    #print (data)
    data = np.vstack( [data,[1,1]])
    data = data[data[:, 0].argsort()]
    data = np.hstack((data, np.zeros((data.shape[0], 1), dtype=data.dtype)))
    data2=data[:25,:].copy()
    
    slopes = []
    
    for i in data[25:,:]:
        #print (data2)
        slopes, interc =QLRegression2( data2, quantile,a,b)
        i[2]=slopes
        print(i)
        data2 = np.vstack( [data2,i])
        
    fig_supl_7=plt.figure(15)
    
    plt.scatter(data[:, 0],data[:, 2],color="r",alpha=0.5)
    plt.xlabel("Human' (days after fertilization) ")
    plt.ylabel("slope of quantile regression ")
    plt.axvline(28, color='red', linewidth=2)
    plt.axvline(280, color='red', linewidth=2)
    #plt.savefig('fig supl 7.png')
    fig_supl_7.savefig('fig supl 6.tif', format='tif', dpi=450)
    