# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 13:45:46 2022

@author: acuaman
"""
import pandas
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
import seaborn as sns
from nonlinearreg import logreg1
from nonlinearreg import func1
import sys
def sim(Min, Max):
    x_all = np.arange(Min, Max, 10)
    y=func1( x_all,10.63536092, -26.82029894)
    y= np.stack((x_all, y,y) , axis=1)
    
    
    return y

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

def coef_var():
    
# def interpolate(x,y):
#     x_all[:int(grp.iloc[:,0].min())]=np.nan
#     x_all = np.arange(x.min(),x.max())
#     arr=[x_all]
#     for i in y:
#         #print (i)
#         interfunction = interp1d(x, i,bounds_error=False, assume_sorted =False)
#         y_all=interfunction(x_all)
#         arr.append(y_all)
        
#     return arr
# def interpolate2(df):
#     x=df.iloc[:,0]
#     x=df.iloc[:,1]
    
# x_all = np.arange(0,10)
# x_all=np.delete(x_all, 3)
# #print (x_all)
# y=np.arange(0,10)
# y1=[np.delete(y, 3)]
# y2=[np.delete(y, 5)]
# lista=[y1,y2]
# print(interpolate(x_all,lista))
def contiguous1(df,data_puntual,rodent='Rat',parameter='Brain Weight'):
    
    

    dfc = df.copy(deep=True)
    dfc['Value'] = pandas.to_numeric(dfc['Value'], errors='coerce')
    dfc = dfc[dfc['Value'].notna()]
    dfc = dfc.sort_values("Pcdays")#groupbw.sort_values(by=["Pcdays"], ascending=True, inplace=True,ignore_index=True)
    pandas.to_numeric(dfc['%'])
    dfc = dfc[["Parameter","Species","Pcdays",'%', "Repeated"]]
    
    figure5=plt.figure(9,figsize=(11.69,8.27))
    spec5 = figure5.add_gridspec(3, 2)
    
    ###############################'Brain Weight'
    dfc_BW= dfc.loc[(dfc["Parameter"] == 'Brain Weight')]
    dfc_H = dfc_BW.loc[(dfc_BW["Species"] == 'Human')]
    dfc_R = dfc_BW.loc[(dfc_BW["Species"] == 'Rat')]
    interfunction1 = interp1d(dfc_H["Pcdays"] ,dfc_H['%'],bounds_error=False, assume_sorted =False)

    hxy=repeted(dfc_H)
    rxy=repeted(dfc_R)
        # GAD activity in cerebellum
        # GAD activity in cortex
        # Somites
    figure5 .add_subplot(spec5[0,0])
    
    
    ax1=sns.lineplot(data=dfc_H , x="Pcdays", y='%',hue = "Repeated",linestyle='--',legend=False)
    ax1.set(xlabel='Human (DAF)', ylabel='% of adult \n brain weight')
    plt.plot(hxy[:,0],hxy[:,1])
    plt.xlim(0,3000)
    plt.axvline(milestones[0][1], color='r')
    # plt.plot(grp.iloc[:,0], grp['%'], ls='--', label=i )
    # plt.xlabel(grp.columns[0])
    # plt.ylabel(key+" (% of adult values)")
    # plt.legend(loc='upper left')#human percentage of increase
    figure5 .add_subplot(spec5[0,1])
    ax2=sns.lineplot(data=dfc_R,x="Pcdays", y='%',hue="Repeated",linestyle='--',legend=False)
    ax2.set(xlabel='Rat (DAF)',ylabel=None)
    plt.plot(rxy[:,0],rxy[:,1])
    plt.xlim(0,150)
    plt.axvline(milestones[1][1], color='r')
    interfunction1 = interp1d(rxy[:,1] ,rxy[:,0],bounds_error=False, assume_sorted =False)
    
    array = np.column_stack((hxy, interfunction1(hxy[:,1])))
    # print (hxy)
    #print(array[:50,:]) 
    # interfunction2 = interp1d(ry, rx ,bounds_error=False, fill_value=0, assume_sorted =False)
    # array2 [:,0]  = hx
    # array2 [:,1] = interfunction2(hy)
    # array= np.concatenate((array1, array2), axis=0)
    figure5 .add_subplot(spec5[1,:])
    plt.plot(array[:,0],array[:,2])
    a=10
    for i in array:
        if i[1]>a:
            plt.plot(i[0],i[2],'o' ,color='cyan', alpha=0.5 )
            plt.text(i[0],i[2],str(a)+'%')
            a=a+20
    
    
    plt.xlim(0, 2000)
    plt.ylim(0, 110) 
    plt.xlabel("Human (DAF)")
    plt.ylabel("Rat (DAF)")
    plt.legend(loc='upper left')
    plt.axvline(milestones[0][1], color='r')
    plt.axhline(milestones[1][1], color='r') 
    #figure5 .add_subplot(spec5[2,0])
    if data_puntual.any():
        #print(data_puntual)
        #model=sim(min(data_puntual[:,0]), max(data_puntual[:,0]))
        interfunction = interp1d(array[:,0],array[:,2],bounds_error=False, assume_sorted =False)
        y_pred = interfunction(data_puntual[:,0])
        array2= np.stack((data_puntual[:,1], y_pred.T) , axis=1)
        #print(array2)
        indexList = [np.any(i) for i in np.isnan(array2)]
        #print(indexList)
        array2 = np.delete(array2, indexList, axis=0)
        R_squared=r2_score(array2[:,0], array2[:,1])
        print('R-squared of'+ 'key'+ ' is'+str(R_squared))
        #print('R-squared of the model in this interval  is'+str(R_squared2))
        figure5 .add_subplot(spec5[2,0])
        plt.plot(data_puntual[:,0],data_puntual[:,1],'o' ,c='#2CBDFE',label='key', alpha=0.5)
        plt.plot(array[:,0],array[:,2],label='key')
        plt.xlim(0, 800)
        plt.ylim(0,60)
        plt.xlabel("Human (DAF)")
        plt.ylabel("Rat (DAF)")
    figure5 .add_subplot(spec5[2,1])
    plt.plot(array[:,0],array[:,2],label='key')
    plt.xlim(0, 800)
    plt.ylim(0,60)
    plt.xlabel("Human (DAF)")
    plt.ylabel("Rat (DAF)")
    plt.savefig('figure 5.png')
    #sys.exit()
    
    if parameter != 'Brain Weight':
        ###############################'GAD activity in cerebellum'
        dfc_GADC= dfc.loc[(dfc["Parameter"] == 'GAD activity in cerebellum')]
        dfc_GADCH = dfc_GADC.loc[(dfc_GADC["Species"] == 'Human')]
        dfc_GADCR = dfc_GADC.loc[(dfc_GADC["Species"] == 'Rat')]
        interfunction3 = interp1d(dfc_GADCH["Pcdays"] ,dfc_GADCH['%'],bounds_error=False, assume_sorted =False)
        interfunction4 = interp1d(dfc_GADCR["Pcdays"] ,dfc_GADCR['%'],bounds_error=False, assume_sorted =False)
        array2 = np.column_stack((hxy, interfunction1(hxy[:,1])))
        figure6=plt.figure(10,figsize=(11.69,8.27))
        spec6 = figure6.add_gridspec(3, 2)
        
        figure6 .add_subplot(spec5[0,0])
        ax3=sns.lineplot(data=dfc_GADCH , x="Pcdays", y='%',hue = "Repeated",linestyle='--',legend=False)
        ax3.set(xlabel='Human (DAF)', ylabel='GAD activity in cerebellum')
        plt.plot(dfc_GADCH["Pcdays"],dfc_GADCH['%'])
        plt.xlim(0,2000)
        plt.axvline(milestones[0][1], color='g')
        figure6 .add_subplot(spec5[0,1])
        ax4=sns.lineplot(data=dfc_GADCR,x="Pcdays", y='%',hue="Repeated",linestyle='--',legend=False)
        ax4.set(xlabel='Rat (DAF)',ylabel=None)
        plt.plot(dfc_GADCR["Pcdays"],dfc_GADCR['%'])
        plt.xlim(0,150)
        plt.axvline(milestones[1][1], color='g')
        figure6 .add_subplot(spec5[2,:])
        interfunction1 = interp1d(dfc_GADCR['%'] ,dfc_GADCR["Pcdays"],bounds_error=False, assume_sorted =False)
        
        dfc_GADCH["interp_Pcdays"] =  interfunction1(dfc_GADCH['%'])
        plt.plot(dfc_GADCH["Pcdays"],dfc_GADCH["interp_Pcdays"])
        a=10
        for i in array:
            if i[1]>a:
                plt.plot(i[0],i[2],'o' ,color='cyan', alpha=0.5 )
                plt.text(i[0],i[2],str(a)+'%')
                a=a+20
        
        
       
            #
        
