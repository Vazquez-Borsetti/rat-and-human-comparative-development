# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 10:37:29 2021

@author: deses
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import QuantileRegressor
import seaborn as sns
import pandas as pd
from nonlinearreg import func1

def QLRegression(data):
    x=np.log(data[:,0])
    y=data[:,1]
    print(x,y)
    mod = smf.quantreg(y,x)
    res = mod.fit(q=0.5)
    print(res.summary())
    



 
def QLRegression2(data, quantile, a,b):
    
    
    x=func1(data[:,0],a,b)
    y=data[:,1]
    x = x[:, np.newaxis]
    qreg = QuantileRegressor(quantile=quantile,alpha=0).fit(x, y)
    slope= qreg.coef_
    intercept = qreg.intercept_
    backlog=np.linspace(1, max(data[:,0]), 10000)
    
    backlog=backlog[:, np.newaxis]
    backlogy=qreg.predict(func1(backlog,a,b))
    
    y_pred  = qreg.predict(x)
    
    
    plot2 = plt.figure(2)


    plt.plot(x, y_pred,c='#F5B14C', label=f"Quantile: {quantile}")
    
    plt.scatter(x,y,c='#2CBDFE',alpha=0.5)
    milestones =np.array([[1,1],[280,22]])
    
    plt.scatter(func1(milestones[:,0],a,b),milestones [:,1]) 
    #plt.scatter(np.log(milestones[2]),milestones[3])
    #backlogx=  data[:,0][:, np.newaxis]
    # backlogx=np.linspace(start = 0, stop = 6600, num = 5)
    # backlogy=  qreg.predict(backlogx)
    plt.xlabel("Human' ")
    plt.ylabel("Rat (days after fertilization)")
    plt.legend(loc='upper left')
    plt.xlim(0)
    plt.ylim(0)
    plot1 = plt.figure(1)
    
    plt.plot(backlog, backlogy,label=f"Quantile: {quantile}")
    plt.scatter(data[:,0],data[:,1],c='#2CBDFE',alpha=0.5)
    plt.scatter(milestones[:,0],milestones [:,1],c='r') 
    
    plt.xlabel("Human (days after fertilization)")
    plt.ylabel("Rat (days after fertilization)")
    plt.legend(loc='upper left')
    plt.xlim(0)
    plt.ylim(0)
    # plot3 = plt.figure(3)
    # sns.lineplot(x, y_pred, label="Quantile: 0.5")
    # sns.lineplot(x, y_predh, label="Quantile: 0.95")
    # sns.lineplot(x, y_predl, label="Quantile: 0.05")
    # sns.scatter(x,y,color="b",alpha=0.5)
    
    # print('****************',y_pred)
    # sns.scatter(np.log(milestones[0]),milestones [1]) 
    # sns.scatter(np.log(milestones[2]),milestones[3])
    plt.show()
    return slope, intercept 

def QLRegression_df(data, quantile, a,b,quant_df,rodent='Rat'):
    
    
    x=func1(data[:,0],a,b)
    x2=func1(quant_df['Human'],a,b)
    y=data[:,1]
    x = x[:, np.newaxis]
    x2 = x2[:, np.newaxis]
    lower_quant2= np.full((len(quant_df[rodent])), 0.0)
    for i in quantile:
        qreg = QuantileRegressor(quantile=i,alpha=0).fit(x, y)
        # slope= qreg.coef_
        # intercept = qreg.intercept_
        # backlog=np.linspace(1, max(data[:,0]), 10000)
        
        # backlog=backlog[:, np.newaxis]
        # backlogy=qreg.predict(func1(backlog,a,b))
        
        y_pred  = qreg.predict(x2)
        
        
        print(y.shape)
        lower_quant = quant_df[rodent] > y_pred 
        lower_quant2[lower_quant] = i
        print(lower_quant2)
    quant_df['quantil'] = lower_quant2
        
        # quant_matrix =  np.vstack((data[:,0],y,y_pred,uper_quant)).transpose()
        
        
    quant_df.to_excel('file2.xlsx')
    
    
    # # construct dataframe, index [0] to make 2d
    # df = pd.DataFrame(quant_matrix)
    
    # save to Excel, exclude index and headers
def QLReg_df_test(data, quantile, a,b,quant_df, ):
    
    x=func1(data[:,0],a,b)
    y=data[:,1]
    x = x[:, np.newaxis]
    lower_quant2= np.full((len(y)), 0.0)
    for i in quantile:
        qreg = QuantileRegressor(quantile=i,alpha=0).fit(x, y)
        # slope= qreg.coef_
        # intercept = qreg.intercept_
        # backlog=np.linspace(1, max(data[:,0]), 10000)
        
        # backlog=backlog[:, np.newaxis]
        # backlogy=qreg.predict(func1(backlog,a,b))
        
        y_pred  = qreg.predict(x)
        
        
        print(y.shape)
        lower_quant = quant_df['Rat'] > y_pred 
        lower_quant2[lower_quant] = i
        print(lower_quant2)
    quant_df['quantil'] = lower_quant2
        
        # quant_matrix =  np.vstack((data[:,0],y,y_pred,uper_quant)).transpose()
        
        
    quant_df.to_excel('file.xlsx')
    
    
    # # construct dataframe, index [0] to make 2d
    # df = pd.DataFrame(quant_matrix)
    
    # save to Excel, exclude index and headers