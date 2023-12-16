import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns
from nonlinearreg import logreg1,logreg3alt
from quantileregresion5 import QLRegression2
from quantileregresion5 import QLRegression_subplot
from bootstraper import bootstrapNLR
from nonlinearreg import func1
from olsreg1 import nlrsm, ols
import statsmodels.api as sm
np.set_printoptions(suppress=True)

palette = ['orange','dodgerblue','darkorchid', 'green', 'magenta','#2CBDFE', 'red','cyan']
plt.style.use('seaborn-whitegrid')
# figure size in inches
#rcParams['figure.figsize'] = 20
plt.rcParams['figure.constrained_layout.use'] = True
pd.options.display.max_columns = None
pd.options.display.max_colwidth = None
pd.options.display.max_rows = None

df=pd.read_excel('age_rat_vs_human_15_11_2023.xlsx')
print(df.head())
campos_iorii=df[df['Author'] =='campos-iorii']
Godlewski=df[df['Author'] =='Godlewski']
Clancy=df[df['Author'] =='Clancy']
Ohmura=df[df['Author'] =='Ohmura']

All=df

def ranged(df,a,b, graph=False,rodent='Rat'):
    # returns a numpy matrix with Start and End points and optionaly a graph

    df2 = df.copy()
    df2 = df2.loc[(df2["Value"] == 'Start') | (df2["Value"] == 'End')]
    
    group = df2.pivot(index=['Parameter', "Value"],
                      columns='Species', values="Pcdays")#.dropna()
    
    group.reset_index(level=None, inplace=True)
    
    if graph:
        milestones =np.array([[1,1],[268,22]])
        figure4=plt.figure(7,figsize=(11.69,8.27))
        

        spec4 = figure4.add_gridspec(2, 2)
        figure4a=figure4 .add_subplot(spec4[0,0])
        
        sns.lineplot(data=group, x='Human', y=rodent,
                     hue='Parameter',legend=False)  # , hue='Parameter'
        sns.scatterplot(data=group, x='Human', y=rodent, hue='Parameter',legend=False)
        figure4a.set_title('a', fontsize='small', loc='left')
        plt.xlim(0,1000)
        plt.ylim(0)
        figure4b=figure4 .add_subplot(spec4[0,1])
        
        group['func human'] = func1(group['Human'],a,b)
        sns.lineplot(data=group, x='func human', y=rodent, hue='Parameter',legend=False)  # , hue='Parameter'
        sns.scatterplot(data=group, x='func human', y=rodent, hue='Parameter',legend=False).set(xlabel='Human\'')
        figure4b.set_title('b', fontsize='small', loc='left')
        
        figure4c=figure4 .add_subplot(spec4[1,:])
        group2 = group.groupby(
            ['Parameter','Value'], axis=0, as_index=True).aggregate('first').unstack()
        
        group2= group2.to_numpy()
        
        
        for i in group2:
            x= np.linspace(i[1], i[0],1000)
            #print(i)
            slope=(i[2]- i[3])/(i[4]- i[5])
            
            #print(slope)
            plt.plot(x,slope* func1(x,a,b)+i[3]-slope*i[5])
        sns.scatterplot(data=group, x='Human', y=rodent, hue='Parameter')
        
        plt.xlim(0,1000)
        plt.ylim(0)
        
        plt.scatter(milestones[:,0],milestones [:,1], c ="pink",linewidths = 2,
                    marker ="s",edgecolor ="green",s = 50)
        figure4c.set_title('c', fontsize='small', loc='left')
        figure4.savefig('fig4.tif', format='tif', dpi=450)
    groupmatrix = group.to_numpy()
    
    return groupmatrix


def puntual(df, pandas_df=False, ranges=False, rodent='Rat' ):
    #returns a numpy array or a df with the developmental milestones of both species as puntual points
    
    if ranges:
        puntual = df.loc[df["Value"].isin(['Puntual','Start','End'])]
        puntual=puntual.sort_values('Pcdays')
       
        puntual['Parameter']= puntual['Parameter'].str.cat(puntual["Value"], sep =", ")
        
    else:
        puntual = df[df.Value == 'Puntual'].sort_values('Pcdays')
    
    puntual2 = puntual[['Parameter', 'Species', "Pcdays"]].groupby(
        ['Parameter', 'Species'], axis=0, as_index=True).aggregate('first').unstack()
    
    
    
    Pcdays = puntual2['Pcdays'].copy()
    Pcdays = Pcdays.dropna(subset=['Human', rodent])
    
    # sys.exit()
    groupmatrix = Pcdays[['Human', rodent]].to_numpy()
    groupmatrix = groupmatrix.astype(float)
 
    if pandas_df:
        groupmatrix = Pcdays[['Human', rodent]]

    return groupmatrix

def rangedcutter(matrix):
    matrixStart = matrix[matrix[:, 1] == 'Start', :]
    matrixEnd = matrix[matrix[:, 1] == 'End', :]
    #print(matrixStart[:, 2:3])
    QLRegression2(matrixEnd[:, 2:4].astype(float))



graph_All = puntual(All, ranges=True ,rodent='Rat')#,graph=False, colors =palette[5]
params = logreg1(graph_All)
with open('databases.txt', 'a') as file:
    file.write(f'campos_iorii rows ={campos_iorii.shape[0]}' )
    file.write(f'Godlewski rows ={Godlewski.shape[0]}'  )
    file.write(f'Clancy rows ={  Clancy.shape[0]}'  )
    file.write(f'Ohmura rows ={ Ohmura.shape[0] }' )
    file.write(f'ALL rows ={ All.shape[0] }' )
    file.write(f'a,b ={ params }' )
print (All.shape)
#%%
#figure 1
figure1=plt.figure(1,constrained_layout=True)#,layout="constrained",figsize=(11.69,8.27)

milestones = [[1, 268], [1, 22]]
graph_CI = puntual(campos_iorii, ranges=True ,rodent='Rat')#
graph_Godlewski= puntual(Godlewski, ranges=True ,rodent='Rat')
graph_clancy= puntual(Clancy, ranges=True ,rodent='Rat')
graph_Ohmura= puntual(Ohmura, ranges=True ,rodent='Rat')

spec1 = figure1.add_gridspec(5, 3)

figure1a=figure1 .add_subplot(spec1[0,0])
graph_CI_pd = puntual(campos_iorii, ranges=True ,pandas_df=True)

sns.scatterplot( data=graph_CI_pd, x=graph_CI_pd['Human'], y=graph_CI_pd["Rat"],color=palette[1], alpha=0.5)
sns.scatterplot(x=milestones[0], y=milestones[1], color='r', alpha=0.5)
figure1a.set_title('a', fontsize='small', loc='left')
figure1 .add_subplot(spec1[0,1])
sns.histplot(graph_CI_pd['Human'], bins=100,
             cumulative=True, element="step", color='aqua')
plt.axvline(milestones[0][1], color='g')

figure1 .add_subplot(spec1[0,2])

sns.histplot(graph_CI_pd['Rat'], bins=100,
             cumulative=True,  element="step", color='tomato')
plt.axvline(milestones[1][1], color='g')
figure1b=figure1 .add_subplot(spec1[1,0])
graph_Godlewski_pd = puntual(Godlewski, ranges=True ,pandas_df=True)

sns.scatterplot( data=graph_Godlewski_pd, x=graph_Godlewski_pd['Human'], y=graph_Godlewski_pd["Rat"],color=palette[2], alpha=0.5)
sns.scatterplot(x=milestones[0], y=milestones[1], color='r', alpha=0.5)
figure1b.set_title('b', fontsize='small', loc='left')
figure1 .add_subplot(spec1[1,1])
sns.histplot(graph_Godlewski_pd['Human'], bins=100,
             cumulative=True, element="step", color='aqua')
plt.axvline(milestones[0][1], color='g')

figure1 .add_subplot(spec1[1,2])

sns.histplot(graph_Godlewski_pd['Rat'], bins=100,
             cumulative=True,  element="step", color='tomato')
plt.axvline(milestones[1][1], color='g')
figure1c=figure1 .add_subplot(spec1[2,0])
graph_Clancy_pd = puntual(Clancy, ranges=True ,pandas_df=True)

sns.scatterplot( data=graph_Clancy_pd, x=graph_Clancy_pd['Human'], y=graph_Clancy_pd["Rat"],color=palette[3], alpha=0.5)
sns.scatterplot(x=milestones[0], y=milestones[1], color='r', alpha=0.5)
figure1c.set_title('c', fontsize='small', loc='left')
figure1 .add_subplot(spec1[2,1])
sns.histplot(graph_Clancy_pd['Human'], bins=100,
             cumulative=True, element="step", color='aqua')
plt.axvline(milestones[0][1], color='g')

figure1 .add_subplot(spec1[2,2])

sns.histplot(graph_Clancy_pd['Rat'], bins=100,
             cumulative=True,  element="step", color='tomato')
plt.axvline(milestones[1][1], color='g')
figure1d=figure1 .add_subplot(spec1[3,0])
graph_Ohm_pd = puntual(Ohmura, ranges=True ,pandas_df=True)

sns.scatterplot( data=graph_Ohm_pd, x=graph_Ohm_pd['Human'], y=graph_Ohm_pd["Rat"],color=palette[4], alpha=0.5)
sns.scatterplot(x=milestones[0], y=milestones[1], color='r', alpha=0.5)
figure1d.set_title('d', fontsize='small', loc='left')
figure1 .add_subplot(spec1[3,1])
sns.histplot(graph_Ohm_pd['Human'], bins=100,
             cumulative=True, element="step", color='aqua')
plt.axvline(milestones[0][1], color='g')

figure1 .add_subplot(spec1[3,2])

sns.histplot(graph_Ohm_pd['Rat'], bins=100,
             cumulative=True,  element="step", color='tomato')
plt.axvline(milestones[1][1], color='g')
figure1e=figure1 .add_subplot(spec1[4,0])
sns.scatterplot( data=graph_CI_pd, x=graph_CI_pd['Human'], y=graph_CI_pd["Rat"],color=palette[1], alpha=0.5)
sns.scatterplot( data=graph_Godlewski_pd, x=graph_Godlewski_pd['Human'], y=graph_Godlewski_pd["Rat"],color=palette[2], alpha=0.8)
sns.scatterplot( data=graph_Clancy_pd, x=graph_Clancy_pd['Human'], y=graph_Clancy_pd["Rat"],color=palette[3], alpha=0.8)
sns.scatterplot( data=graph_Ohm_pd, x=graph_Ohm_pd['Human'], y=graph_Ohm_pd["Rat"],color=palette[4], alpha=0.3)
sns.scatterplot(x=milestones[0], y=milestones[1], color='r', alpha=0.5)
plt.xlim(0, 800)
plt.ylim(0, 50)
figure1e.set_title('e', fontsize='small', loc='left')
figure1 .add_subplot(spec1[4,1])
graph_All_pd = puntual(All, ranges=True ,pandas_df=True)
sns.histplot(graph_All_pd['Human'], bins=100,
              cumulative=True, element="step", color='aqua')
plt.axvline(milestones[0][1], color='g')

figure1 .add_subplot(spec1[4,2])

sns.histplot(graph_All_pd['Rat'], bins=100,
              cumulative=True,  element="step", color='tomato')
plt.axvline(milestones[1][1], color='g')
plt.tight_layout()
figure1.savefig('fig1.tif', format='tif', dpi=600)

#%%
########################################suplementary1 (fig 2 and 3)

ols(graph_CI,params[0],params[1],'graph_CI',colors =palette[1])
ols(graph_Godlewski,params[0],params[1],'graph_Godlewski',colors =palette[2])
ols(graph_clancy,params[0],params[1],'graph_clancy',colors =palette[3])
ols(graph_Ohmura,params[0],params[1],'graph_Ohm',colors =palette[4])
ols(graph_All,params[0],params[1],'graph_All',colors =palette[5])

nlrsm(graph_CI,'graph_CI',params[0],params[1])
nlrsm(graph_Godlewski,'graph_Godlewski',params[0],params[1])
nlrsm(graph_clancy,'graph_clancy',params[0],params[1])
nlrsm(graph_Ohmura,'graph_Ohm',params[0],params[1])
nlrsm(graph_All,'graph_All',params[0],params[1])




#%%
#figure 2,all 4 data bases together 
milestones = [[1, 268], [1, 22]]

figure2=plt.figure(5,figsize=(11.69,8.27))
spec2 = figure2.add_gridspec(2, 2)
figure2a=figure2 .add_subplot(spec2[0,:])
logreg1(graph_All,colors=palette[5], graph=True)
figure2a.set_title('a', fontsize='small', loc='left')
figure2b=figure2 .add_subplot(spec2[1,0])#.margins(x=0, y=-0.25)
plt.xlim(0, 280)
plt.ylim(0, 30) 
logreg1(graph_All,colors=palette[5], graph=True)
figure2b.set_title('b', fontsize='small', loc='left')
figure2c=figure2 .add_subplot(spec2[1,1])#.margins(x=0, y=-0.25) 
logreg1(graph_All,colors=palette[5], graph=True)
figure2c.set_title('c', fontsize='small', loc='left')
plt.xlim(100, 800)
plt.ylim(7, 50) 

figure2.savefig('fig2.tif', format='tif', dpi=450)

#%%
#FIGURE 3
figure3=plt.figure(6,figsize=(11.69,8.27))
print(params)
spec3 = figure3.add_gridspec(3, 3)

tuple_qr_bottom =QLRegression_subplot(graph_All, 0.05,params[0],params[1])
tuple_qr_median =QLRegression_subplot(graph_All, 0.5,params[0],params[1])
tuple_qr_top =QLRegression_subplot(graph_All, 0.95,params[0],params[1])    

table1=np.stack((tuple_qr_bottom[-2], tuple_qr_bottom[-1],tuple_qr_median[-1],tuple_qr_top[-1]), axis=1).reshape(-1,4)
table1 = pd.DataFrame(table1, columns = ['DAF Human','DAF Rat bottom','DAF Rat median','DAF Rat top'])#,'DAF Rat Top'
table1.to_excel('table1.xlsx', sheet_name='new_sheet_name')

figure3a=figure3 .add_subplot(spec3[0,:])
plt.scatter(tuple_qr_bottom[2],tuple_qr_bottom[3],c='#2CBDFE',alpha=0.5)
plt.plot(tuple_qr_bottom[2], tuple_qr_bottom[4],c='y', label="Quantile: 0.05")
plt.plot(tuple_qr_median[2], tuple_qr_median[4],c='g', label="Quantile: 0.5")
plt.plot(tuple_qr_top[2], tuple_qr_top[4],c='b', label="Quantile: 0.95")
milestones =np.array([[1,1],[268,22]])

plt.scatter(func1(milestones[:,0],params[0],params[1]),milestones [:,1],c='r') 
figure3a.set_title('a', fontsize='small', loc='left')
plt.xlabel("Human' ")
plt.ylabel("Rat (days after fertilization)")
plt.legend(loc='upper left')
plt.xlim(0)
plt.ylim(0)

figure3b=figure3 .add_subplot(spec3[1,:])
plt.scatter(graph_All[:,0],graph_All[:,1],c='#2CBDFE',alpha=0.5)
plt.plot(tuple_qr_bottom[-2], tuple_qr_bottom[-1],c='y', label="Quantile: 0.05")
plt.plot(tuple_qr_median[-2], tuple_qr_median[-1],c='g', label="Quantile: 0.5")
plt.plot(tuple_qr_top[-2], tuple_qr_top[-1],c='b', label="Quantile: 0.95")
milestones =np.array([[1,1],[268,22]])

plt.scatter(milestones[:,0],milestones [:,1],c='r') 

figure3b.set_title('b', fontsize='small', loc='left')
plt.xlabel("Human (days after fertilization)")
plt.ylabel("Rat (days after fertilization)")
plt.legend(loc='upper left')
plt.xlim(0)
plt.ylim(0)
figure3c=figure3 .add_subplot(spec3[2,0])
plt.xlim(0, 280)
plt.ylim(0, 30) 
plt.scatter(graph_All[:,0],graph_All[:,1],c='#2CBDFE',alpha=0.5)
plt.plot(tuple_qr_bottom[-2], tuple_qr_bottom[-1],c='y')
plt.plot(tuple_qr_median[-2], tuple_qr_median[-1],c='g')
plt.plot(tuple_qr_top[-2], tuple_qr_top[-1],c='b')
milestones =np.array([[1,1],[268,22]])
figure3c.set_title('c', fontsize='small', loc='left')
plt.scatter(milestones[:,0],milestones [:,1],c='r') 
figure3d=figure3 .add_subplot(spec3[2,1])
plt.scatter(graph_All[:,0],graph_All[:,1],c='#2CBDFE',alpha=0.5)
plt.plot(tuple_qr_bottom[-2], tuple_qr_bottom[-1],c='y')
plt.plot(tuple_qr_median[-2], tuple_qr_median[-1],c='g')
plt.plot(tuple_qr_top[-2], tuple_qr_top[-1],c='b')


plt.scatter(milestones[:,0],milestones [:,1],c='r') 
figure3d.set_title('d', fontsize='small', loc='left')
plt.xlim(100, 800)
plt.ylim(7, 50) 
figure3e=figure3 .add_subplot(spec3[2,2])


plt.scatter(graph_All[:,0],graph_All[:,1],c='#2CBDFE',alpha=0.5)
plt.plot(tuple_qr_median[-2], tuple_qr_median[-1],c='g',label="Quantil reg.")
x_fig3 = np.linspace(1, max(graph_All[:,0]),100000)
plt.plot(x_fig3, func1(x_fig3,params[0],params[1]),c='#2CBDFE',label="Equation 1 fit")

sns.regplot(x= graph_All[:,0],y=graph_All[:,1],scatter=False,lowess=True,label='Lowess reg')
plt.scatter(milestones[:,0],milestones [:,1],c='r') 
figure3e.set_title('e', fontsize='small', loc='left')
plt.xlim(0,750)
plt.ylim(0,60)

plt.legend(loc='upper left')

figure3.savefig('fig3.tif', format='tif', dpi=450)
#%%
#figure 4



ranged(campos_iorii,params[0],params[1],graph=True)

#%%
figsupl4=plt.figure(8)
lowess = sm.nonparametric.lowess(graph_All[:,1],graph_All[:,0], frac=.3)
lowess_x = list(zip(*lowess))[0]
lowess_y = list(zip(*lowess))[1]
f = interp1d(lowess_x, lowess_y, bounds_error=False)
xnew_lowess = np.arange(graph_All[:,0].min(), graph_All[:,0].max())

ynew = f(xnew_lowess)

plt.plot(tuple_qr_median[-2], tuple_qr_median[-1],c='g',label="Quantil reg.")

plt.plot(x_fig3, func1(x_fig3,params[0],params[1]),c='#2CBDFE',label="Equation 2")
plt.plot(graph_All[:,0],graph_All[:,1], 'o')
plt.axvline(268, color='red', linewidth=2)


plt.plot(xnew_lowess, ynew, '-',label='Lowess reg')
plt.legend(loc='upper left')
plt.xlim(0)
plt.ylim(0)
figsupl4.savefig('fig supl 3.tif', format='tif', dpi=450)
plt.show()

################### confidence intervals 
bootstrapNLR(graph_All,5000,colors =palette[0])
figsupl5=plt.figure(9)
params = logreg3alt(graph_All, graph=True)
plt.show()