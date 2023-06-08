# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 13:45:46 2022

@author: PV
"""
import pandas
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from sklearn.metrics import r2_score
import seaborn as sns
from nonlinearreg import logreg1
from nonlinearreg import func1
import sys
# def sim(Min, Max):
#     x_all = np.arange(Min, Max, 10)
#     y=func1( x_all,10.63536092, -26.82029894)
#     y= np.stack((x_all, y,y) , axis=1)
    
    
#     return y

def repeted(group):
    
    pcd=group["Pcdays"]
    
    group= group.groupby(["Repeated"])
    arr = []
    
    for i, grp in group:
        
        
        x_all = np.arange(0,pcd.max())
        
        x_all[:int(grp["Pcdays"].min())]=np.nan
        x_all[int(grp["Pcdays"].max()):]=np.nan
        y_all =x_all.copy()
        # print('int(grp.iloc[:,0].min())')
        # print(int(grp.iloc[:,0].min()))
        interfunction = interp1d(grp["Pcdays"], grp["%"],bounds_error=False, assume_sorted =False)#
        #y_all=interfunction(x_all[int(grp.iloc[:,0].min()):int(grp.iloc[:,0].max()):])
        y_all[int(grp["Pcdays"].min()):int(grp["Pcdays"].max())]=interfunction(x_all[int(grp["Pcdays"].min()):int(grp["Pcdays"].max())])#data1 = data 
        
        #y_all=interfunction(x_all[int(grp.iloc[:,0].min()):int(grp.iloc[:,0].max()):])
        
        #print(y_all[int(grp["Pcdays"].min())])
        # print(y_all)
        arr.append(y_all)
        
        # print(grp['%'])
        
    arr = np.array(arr)
    arr=np.transpose(arr)
    #print (arr)
    # 
    
    arr=np.nanmean(arr, axis=1)
    array= np.stack((np.arange(0,pcd.max()), arr.T) , axis=1)#pcd.min()
    #array=array[~np.isnan(array).any(axis=1), :]
    
    #plt.plot(array[:,0], array[:,1], ls='-', label='mean' )
    
    return array

def contiguous1(df,parameter='Brain Weight'):
    
    np.set_printoptions(suppress=True)
    

    dfc = df.copy(deep=True)
    dfc['Value'] = pandas.to_numeric(dfc['Value'], errors='coerce')
    dfc = dfc[dfc['Value'].notna()]
    dfc = dfc.sort_values("Pcdays")#groupbw.sort_values(by=["Pcdays"], ascending=True, inplace=True,ignore_index=True)
    pandas.to_numeric(dfc['%'])
    dfc = dfc[["Parameter","Species","Pcdays",'%', "Repeated"]]
    
    dfc_BW= dfc.loc[(dfc["Parameter"] == parameter)]
    dfc_H = dfc_BW.loc[(dfc_BW["Species"] == 'Human')]
    dfc_R = dfc_BW.loc[(dfc_BW["Species"] == 'Rat')]
    dfc_H2=repeted(dfc_H)
    dfc_R2=repeted(dfc_R)
    
    return dfc_H ,dfc_H2,dfc_R,dfc_R2
