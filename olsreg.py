# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 12:22:59 2022

@author: acuaman
"""

from __future__ import print_function
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)
import pandas as pd
import seaborn as sns
from nonlinearreg import func1
import statsmodels.api as sm

import statsmodels.formula.api as smf

import numpy as np

import pandas

def func3(x):
     return np.log(x ) 

def nlrsm(data,name):
    df = pandas.DataFrame(data, columns=['xData','yData'])
    formula = "yData ~ func3(xData)"
    mod = smf.ols(formula , data=df)
    
    res = mod.fit()
    
    #print(res.summary())
    file_object = open('statsnl.txt', 'a')
    file_object.write(name)
    file_object.write(str(res.summary()))
    
    file_object.close()

def ols(data,a,b, name,colors='b'):
    xData,yData= data[:,0],data[:,1]
    
    #xData = sm.add_constant(xData)# adding the constant term
    result = sm.OLS(yData,xData).fit()
    
    #print(result.summary())
    # fig = plt.figure(figsize=(14, 8))
    # fig = sm.graphics.plot_regress_exog(result,np.ndindex(data),fig=fig)
    plt.figure(2)
    sns.residplot(xData,yData, lowess=True, color=colors,label=name)
    plt.xlim(0, 800)
    
    plt.show()
    plt.savefig('fig supl 1.png')
    plt.figure(3)
    plt.savefig('fig supl 2.png')
    x=func1(xData,a,b)
    result2 = sm.OLS(yData,x).fit()
    #rint(result2.summary())
    sns.residplot(x,yData, lowess=True, color=colors)
    #print(result.summary())
    file_object = open('statsolslr.txt', 'a')
    file_object.write(name)
    file_object.write(str(result2.summary()))
def wls(data):
    xData,yData= data[:,0],data[:,1]
    exog = sm.add_constant(xData)
    print(yData.shape) 
    w = np.ones(yData.shape[0])
    weights = sm.WLS(yData, exog, weights=1./(w ** 2)) # Calculating the weights
    wls = sm.WLS(yData, exog, weights)
    results = wls.fit(cov_type="fixed scale")
    print(results.summary())

