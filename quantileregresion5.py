# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 10:37:29 2021

@author: deses
"""
import numpy as np


import statsmodels.formula.api as smf


from sklearn.linear_model import QuantileRegressor

from nonlinearreg import func1

def QLRegression(data):
    x=np.log(data[:,0])
    y=data[:,1]
    #print(x,y)
    mod = smf.quantreg(y,x)
    res = mod.fit(q=0.5)
    print(res.summary())

def QLRegression2(data,quantile, a,b ):
    
    
    x=func1(data[:,0],a,b)
    y=data[:,1]
    x = x[:, np.newaxis]
    qreg = QuantileRegressor(quantile=quantile,alpha=0,solver='highs').fit(x, y)
    slope= qreg.coef_
    intercept = qreg.intercept_
    backlog=np.linspace(1, max(data[:,0]), 10000)
    
    backlog=backlog[:, np.newaxis]
    backlogy=qreg.predict(func1(backlog,a,b))
    
    y_pred  = qreg.predict(x)
    
    
    return slope, intercept 

def QLRegression_subplot(data,quantile, a,b):
    
    
    x=func1(data[:,0],a,b)
    y=data[:,1]
    x = x[:, np.newaxis]
    qreg = QuantileRegressor(quantile=quantile,alpha=0).fit(x, y)
    slope= qreg.coef_
    intercept = qreg.intercept_
    max_days=int(max(data[:,0]))
    backlog=np.linspace(1,max_days , max_days)
    
    backlog=backlog[:, np.newaxis]
    backlogy=qreg.predict(func1(backlog,a,b)).reshape(-1,1)
    
    y_pred  = qreg.predict(x)#
    return slope, intercept,x,y, y_pred,backlog, backlogy
   
def QLRegression_df(data, quantile, a,b,quant_df,rodent='Rat'):
    
    
    x=func1(data[:,0],a,b)
    x2=func1(quant_df['Human'],a,b)
    y=data[:,1]
    x = x[:, np.newaxis]
    x2 = x2[:, np.newaxis]
    lower_quant2= np.full((len(quant_df[rodent])), 0.0)
    for i in quantile:
        qreg = QuantileRegressor(quantile=i,alpha=0.5).fit(x, y)
       
        y_pred  = qreg.predict(x2)
        
        
        print(y.shape)
        lower_quant = quant_df[rodent] > y_pred 
        lower_quant2[lower_quant] = i
        #print(lower_quant2)
    quant_df['quantil'] = lower_quant2

    quant_df.to_excel('file2.xlsx')
    
 