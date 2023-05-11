# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 20:29:56 2021

@author: deses
"""



import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import r2_score


def func1(x, a, b):
     return a * np.log(x )+b
def func2logy(data):
    data[:,1]=np.log(data[:,1]+0.0001)
    return data

def logreg1(data, colors='#2CBDFE', graph=False):
    # non linear regression using ordinary least squares from scipy
    
    xData,yData= data[:,0],data[:,1]
    popt, pcov = scipy.optimize.curve_fit(func1, xData, yData, maxfev=10000)#,p0=[0.1,0.1]
    # print (np.sqrt(np.diag(pcov)))#standar deviation of the parameters
    # print(popt)
    x = np.linspace(1, max(xData),100000)
    #calculating R2
    y_pred = func1(xData, *popt)
    # print('R-squared='+str(r2_score(yData, y_pred)))
    milestones =[[1,280],[1,22]]
    if graph:
        plt.figure(5)
        
        plt.scatter(xData, yData,c='#2CBDFE', label='Dev milestones', marker='o', alpha=(0.5))
        plt.scatter( milestones[0],milestones[1],c='red', marker='o')
        plt.plot(x, func1(x,popt[0],popt[1]),color=colors , label="Log Fit",alpha=(0.9))
        plt.xlabel("Human (days after fertilization)")
        plt.ylabel("Rat (days after fertilization)")
        #plt.legend(loc='upper left')
        plt.ylim(ymin=0)
        plt.show()
    return popt

def logreg2(data):
    # non linear regression using ordinary least squares from scipy
    
    xData,yData= data[:,0],data[:,1]
    popt, pcov = scipy.optimize.curve_fit(func1, xData, yData, maxfev=10000)#,p0=[0.1,0.1]
    #print (popt)
    x = np.linspace(1, max(xData),100000)
    #calculating R2
    y_pred = func1(xData, *popt)
    # print('R-squared='+str(r2_score(yData, y_pred)))
    milestones =[[1,280],[1,22]]
    array= np.stack((x, func1(x,popt[0],popt[1])) , axis=1)
    plt.figure(1)
    
    plt.scatter(xData, yData,c='#2CBDFE', label='Dev milestones', marker='o', alpha=(0.5))
    plt.scatter( milestones[0],milestones[1], marker='o')
    plt.plot(x, func1(x,popt[0],popt[1]), '#2CBDFE', label="Log Fit",alpha=(0.9))
    plt.xlabel("Human (days after fertilization)")
    plt.ylabel("Rat (days after fertilization)")
    #plt.legend(loc='upper left')
    plt.ylim(ymin=0)
    plt.show()
    return array
    