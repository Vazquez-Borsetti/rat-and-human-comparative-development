# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 09:33:19 2022

@author: deses
"""

import numpy as np
from quantileregresion5 import QLRegression2
from bootstraper import bootstrap
import matplotlib.pyplot as plt
import seaborn as sns

def splines(data,quantile,cut_date, n_boots,a,b):
    #print (data)
    
    
    data1 = data [data[:,0]< cut_date]
    data2 = data [data[:,0]> cut_date]
    slopes1, interc1 =bootstrap( data1, quantile,n_boots,a,b)
    slopes2, interc2 =bootstrap( data2, quantile,n_boots,a,b)
    slop_dif=slopes1-slopes2
    intersec_dif=interc1-interc2
    print('slope diference')
    print (slop_dif, np.percentile(slop_dif, 2.5),np.percentile(slop_dif, 97.5))
    print ('intersec diference')
    print (intersec_dif, np.percentile(intersec_dif, 2.5),np.percentile(intersec_dif, 97.5))
        
    plot5= plt.figure(5)
    
    plt.grid(True)
    
    plt.title('slope diference')
    
    sns.distplot(slop_dif)
    
    plt.axvline(np.percentile(slop_dif, 2.5), color='red', linewidth=2) ;
    
    plt.axvline(np.percentile(slop_dif, 97.5), color='red', linewidth=2) ;
    