# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 15:24:55 2022

@author: acuaman
"""
import colorama
from colorama import Fore, Style
import matplotlib.pyplot as plt

import pandas




def frequency(data):
    
    plt.figure(4)
    data = data.loc[data['Species'] == 'Human' and data['Species'] == 'Rat', 'A']
    data2 = data.loc[data['Species'].isin(['Human','Rat'])]
    totalrows=data2['author'].value_counts()#[:20].plot(kind='barh')
    datapuntual = data2.loc[data2["Value"].isin(['Puntual','Start','End'])]
    
    
    
    totalrows2=datapuntual['author'].value_counts()
    data=datapuntual[['author','DOIs']]
    camposiorii=data.groupby('author').get_group('campos-iorii')['DOIs']
    Clancy=data.groupby('author').get_group('Clancy')['DOIs']
    Ohmura=data.groupby('author').get_group('Ohmura')['DOIs']
    Godlewski=data.groupby('author').get_group('Godlewski')['DOIs']
    
    intersec=set(Clancy) & set(camposiorii)
    #data =pandas.crosstab(index=data.author, columns=data.DOIs)#, aggfunc=count
    #data = pandas.pivot_table(data, index=['DOIs'] , aggfunc='size')
    #data=pandas.crosstab(datapuntual.DOIs,datapuntual.author)
    print('******************')
    print(intersec)
    #data2=data.set_index(['DOIs','author']).unstack(level=0)
    #data2 =data.groupby(['author','DOIs']).value_counts().unstack()
    #.pivot(columns='DOIs', values='author').reset_index()
    # data = pandas.crosstab(data['author'],data['DOIs'])
    # data.plot(kind='bar',  rot=0)
    # for i in list_data:
    #     print(i.head())
        
    #     puntual = i.loc[i["Value"].isin(['Puntual','Start','End'])]
    
    #     list_of_df.append (puntual['DOIs'])
    print(Fore.GREEN + 'DOIS')
    # print(list_of_df.shape)
    # print(list_of_df)
    # dois =puntual.groupby('author')['DOIs'].value_counts()
    # plt.hist(total,bins=14)
    # plt.show()
    print ("total rows")
    print(totalrows)
    print ("total rows (puntual)")
    print(totalrows2)
    
    #print(data.head(-10))
    print(Style.RESET_ALL)