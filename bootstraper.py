# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 21:33:17 2021

@author: deses
"""
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm
import seaborn as sns
from quantileregresion5 import QLRegression2
from  nonlinearreg import logreg1
# synthetic sample data
def bootstrap(data,quantile,n_boots, a,b):
    
    # resample with replacement each row
    boot_slopes = []
    boot_interc = []
    params=[]
    plt.figure(1)
    for _ in range(n_boots):
        # sample the rows, same size, with replacement
        sample_df = sklearn.utils.resample(data, replace=True)
        # fit a  regression( linear, qunatile, etc)
        slopes, interc =QLRegression2( sample_df, quantile,a,b)
       
     
     # ols_model_temp = sm.ols(formula = 'y ~ x', data=sample_df)
     # results_temp = ols_model_temp.fit()
     
        # append coefficients
        boot_slopes.append(slopes)
        boot_interc.append(interc)
        
    boot_slopes = np.array(boot_slopes)
    boot_interc= np.array(boot_interc)
    # print('slope')
    # print (boot_slopes, np.percentile(boot_slopes, 2.5),np.percentile(boot_slopes, 97.5))#95% CI mil resamples 7.213475187696851-11.02952096396685, +wanda 7.90873488240883 10.050061325615975
    # print ('intersec')
    # print (boot_interc, np.percentile(boot_interc, 2.5),np.percentile(boot_interc, 97.5))
        #y_pred_temp = ols_model_temp.fit().predict(sample_df['x'])
        #plt.plot(sample_df['x'], y_pred_temp, color='grey', alpha=0.2)
    # add data points
    
    #plt.plot(x, y_pred, linewidth=2)
    figure_supl_5=plt.figure(13,figsize=(100,50))
    
    plt.grid(True)
    
    plt.title('slopes')
    
    sns.distplot(boot_slopes)
    
    plt.axvline(np.percentile(boot_slopes, 2.5), color='red', linewidth=2) ;
    
    plt.axvline(np.percentile(boot_slopes, 97.5), color='red', linewidth=2) ;
    figure_supl_5=plt.figure(14,figsize=(100,50))
    
    plt.grid(True)
    
    
    plt.title('interc')
    
    sns.distplot(boot_interc)
    
    plt.axvline(np.percentile(boot_interc, 2.5), color='red', linewidth=2) ;
    
    plt.axvline(np.percentile(boot_interc, 97.5), color='red', linewidth=2) ;
    
    
    plt.show()
    return boot_slopes, boot_interc

def bootstrapNLR(data, n_boots, fig='', colors='#2CBDFE'):
    
    # resample with replacement 
    
    a=[]
    b=[]
    plt.figure(1)
    for _ in range(n_boots):
        # sample, same size, with replacement
        sample_df = sklearn.utils.resample(data, replace=True)
        # fit the regression
        params = logreg1(sample_df)
        a.append(params[0])
        b.append(params[1])
    a = np.array(a)
    b= np.array(b)
   
        #y_pred_temp = ols_model_temp.fit().predict(sample_df['x'])
        #plt.plot(sample_df['x'], y_pred_temp, color='grey', alpha=0.2)
    # add data points
    
    #plt.plot(x, y_pred, linewidth=2)
    fig_supl_5=plt.figure(13,figsize=(11.69,8.27))
    
    plt.grid(True)
    
    plt.title('a')
    
    sns.distplot(a,color=colors)
    
    plt.axvline(np.percentile(a, 2.5), color=colors, linewidth=1, alpha=1) 
    
    plt.axvline(np.percentile(a, 97.5), color=colors, linewidth=1, alpha=1) 
    fig_supl_5.savefig('fig supl 4.tif', format='tif', dpi=450)
    fig_supl_6=plt.figure(14,figsize=(11.69,8.27))
    
    plt.grid(True)
    
    
    plt.title('b')
    
    sns.distplot(b,color=colors)
    
    plt.axvline(np.percentile(b, 2.5), color=colors, linewidth=1, alpha=1) 
    
    plt.axvline(np.percentile(b, 97.5), color=colors, linewidth=1, alpha=1) 
    with open('databases.txt', 'a') as file:
        file.write(f'a confidence 95% ={np.percentile(a, 2.5),np.percentile(a, 97.5)}' )
        file.write(f'b confidence 95% ={np.percentile(b, 2.5),np.percentile(b, 97.5)}' )
    fig_supl_6.savefig('fig supl 5.tif', format='tif', dpi=450)
    plt.show()
    return a, b