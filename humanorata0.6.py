import pandas
import math
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
from cycler import cycler
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from tempfile import TemporaryFile
from nonlinearreg import logreg1
from quantileregresion4 import QLRegression2
from quantileregresion4 import QLRegression_df
from quantileregresion4 import func1
from bootstraper import bootstrap, bootstrapNLR
from sloper import slopes
from patsy import dmatrix
from spliner import splines
sns.set()
sns.set_theme(style="darkgrid")
plt.style.use('ggplot')


# reads exel with data
#dfa = pandas.read_excel('DB Antonella.xlsx', sheet_name='Hoja 2')
#dfc = pandas.read_excel('human-rat-comparisontomas.xlsx', sheet_name='Hoja 1')
df = pandas.read_excel('Human-Rat_comparison-DB-wanda.xlsx', sheet_name='Sheet1')
#df = pandas.concat([dfa, dfb, dfc], ignore_index=True).reset_index()
pandas.to_numeric(df["Pcdays"])
df = df.filter(items=["Value", 'Species', 'Parameter', "Pcdays"])

# buscador de duplicados
# duplicates = df[df.duplicated()]
# print("Duplicates :")
# print(duplicates)

pandas.options.display.max_columns = None
pandas.options.display.max_colwidth = None
pandas.options.display.max_rows = None


def ranged(df,a,b, graph=False,rodent='Rat'):
    # returns a numpy matrix with raged data and x (human)is log scale

    df2 = df.copy()
    df2 = df2.loc[(df2["Value"] == 'Start') | (df2["Value"] == 'End')]
    # mask = df['Species'] == 'Human'
    # df2.loc[mask, "Pcdays"] = df2.loc[mask,"Pcdays"].apply(func2)

    #sns.pointplot(x='table', y='value', hue='c', data=df_long, join=False, dodge=0.2)
    #print( df2)
    

    group = df2.pivot(index=['Parameter', "Value"],
                      columns='Species', values="Pcdays")#.dropna()
    
    group.reset_index(level=None, inplace=True)
    #print( group)
    if graph:
        plt.figure(10)
        # print(group)
        #sns.lineplot(x='Human', y="Rat", hue='Parameter', data=pandas.melt(group, ['Value']))
        sns.lineplot(data=group, x='Human', y=rodent,
                     hue='Parameter')  # , hue='Parameter'
        sns.scatterplot(data=group, x='Human', y=rodent, hue='Value')
        plt.figure(11)
        
        group['func human'] = func1(group['Human'],a,b)
        sns.lineplot(data=group, x='func human', y=rodent, hue='Parameter')  # , hue='Parameter'
        sns.scatterplot(data=group, x='func human', y=rodent, hue='Value')
        # df2.loc[mask, "Pcdays"] = df2.loc[mask,"Pcdays"].apply(func2)
        
        plt.figure(1)
        group2 = group.groupby(
            ['Parameter','Value'], axis=0, as_index=True).aggregate('first').unstack()
        group2= group2.to_numpy()
        
        
        
        for i in group2:
            x= np.linspace(i[1], i[0],1000)
            print(i)
            slope=(i[4]- i[5])/(i[6]- i[7])
            print(slope)
            plt.plot(x,slope* func1(x,a,b)+i[5]-slope*i[7])
        sns.scatterplot(data=group, x='Human', y=rodent, hue='Parameter')
           
        
        
    groupmatrix = group.to_numpy()
    #print(group['func human'])
    return groupmatrix


def puntual(df, graph=False, pandas_df=False, ranges=False, rodent='Rat'):
    #returns a numpy array with the developmental milestones of both species as puntual points
    
    if ranges:
        puntual = df.loc[df["Value"].isin(['Puntual','Start','End'])]
        puntual=puntual.sort_values('Pcdays')
    else:
        puntual = df[df.Value == 'Puntual'].sort_values('Pcdays')

    puntual2 = puntual[['Parameter', 'Species', "Pcdays"]].groupby(
        ['Parameter', 'Species'], axis=0, as_index=True).aggregate('first').unstack()

    #print (puntual)
    Pcdays = puntual2['Pcdays'].copy()
    Pcdays = Pcdays.dropna(subset=['Human', rodent])

    Pcdays['Human'] = pandas.to_numeric(Pcdays['Human'], downcast="float")
    Pcdays[rodent] = pandas.to_numeric(Pcdays[rodent], downcast="float")
   # print (Pcdays[Pcdays['Rat'].gt(30) & a['Human'].le(200) ])
    #print(Pcdays[['Human', rodent]])
    groupmatrix = Pcdays[['Human', rodent]].to_numpy()
    groupmatrix = groupmatrix.astype(float)
    if rodent=='Rat':
        milestones = [[1, 280], [1, 22]]
    elif rodent=='Mouse':
        milestones = [[1, 280], [1, 20]]

    if graph:
        plt.figure(1)
        sns.scatterplot(
            data=Pcdays, x=Pcdays['Human'], y=Pcdays[rodent], alpha=0.5)
        sns.scatterplot(x=milestones[0], y=milestones[1])
        plot2 = plt.figure(2)
        sns.histplot(Pcdays['Human'], bins=100,
                     cumulative=True, element="step", color='aqua')
        plt.axvline(milestones[0][1], color='g')
        plot3 = plt.figure(3)
        sns.histplot(Pcdays[rodent], bins=100,
                     cumulative=True,  element="step", color='tomato')
        plt.axvline(milestones[1][1], color='g')

        plt.show()
    if pandas_df:
        groupmatrix = Pcdays[['Human', rodent]]
        # print(groupmatrix['Rat'])

    return groupmatrix

def rangedcutter(matrix):
    matrixStart = matrix[matrix[:, 1] == 'Start', :]
    matrixEnd = matrix[matrix[:, 1] == 'End', :]
    print(matrixStart[:, 2:3])
    QLRegression2(matrixEnd[:, 2:4].astype(float))


graph = puntual(df, ranges=True ,graph=False,rodent='Rat')

#params = logreg1(graph)
#print(ranged(df,params[0],params[1]))
#quant_df = puntual(df, pandas_df=True)
#graph=np.concatenate(puntual(df), ranged(df,params[0],params[1]))

print(graph.shape)
graph = graph [graph[:,0]< 6000]
params = logreg1(graph)
print(graph.shape)
print(params)
#QLRegression2(graph, 0.5,params[0],params[1])
for i in [0.05,0.5,0.95]:
      QLRegression2(graph, i,params[0],params[1])
    
# rangedcutter(ranged(df))


#QLRegression_df(graph, [0.1  , 0.9],params[0],params[1],quant_df)
# logreg2(graph)

# graph[np.all(graph != 0, axis=1)]
#slopes(graph, 0.5,params[0],params[1])
#bootstrapNLR(graph,5000)
#bootstrap(graph, 0.5,100,params[0],params[1])
#splines(graph, 0.5,280, 300,params[0],params[1])
#bootstrap(graph, 0.5,100,params[0],params[1])
# ax = sns.kdeplot(graph[:,0],graph[:,1], shade=True, thresh=0.05, alpha=.3,n_levels=15,cmap='magma')
# ax = sns.scatterplot(graph[:,0],graph[:,1])
# ax = plt.scatter(x=280,y=22,color='r')
