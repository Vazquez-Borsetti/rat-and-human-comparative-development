import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import rcParams
import sys
from scipy.interpolate import interp1d
import seaborn as sns
from nonlinearreg import logreg1
from quantileregresion5 import QLRegression2
from quantileregresion5 import QLRegression_subplot
from contiguous4 import contiguous1
from bootstraper import bootstrap, bootstrapNLR
from sloper import slopes
from spliner import splines
from dois import frequency
from olsreg1 import ols
from nonlinearreg import func1
from olsreg1 import nlrsm
from sklearn.metrics import r2_score
import PIL
np.set_printoptions(suppress=True)

palette = ['orange','dodgerblue','darkorchid', 'green', 'magenta','#2CBDFE', 'red','cyan']
plt.style.use('seaborn-whitegrid')
# figure size in inches
#rcParams['figure.figsize'] = 20
plt.rcParams['figure.constrained_layout.use'] = True
pandas.options.display.max_columns = None
pandas.options.display.max_colwidth = None
pandas.options.display.max_rows = None
def reader(dbx=False):
    # reads exel and gives format to data 
    dfa = pandas.read_excel('Campos-Iorii.xlsx', sheet_name='Sheet1')
    dfa['author'] ='campos-iorii'
    dfb = pandas.read_excel('Godlewski.xlsx', sheet_name='Godlewski')
    dfb['author'] ='Godlewski'
    dfc = pandas.read_excel('Clancy.xlsx', sheet_name='Empíricos')
    dfc['author'] ='Clancy'
    dfd = pandas.read_excel('Ohmura.xlsx', sheet_name='Ohmura')
    dfd['author'] ='Ohmura'
    list_db=[dfa,dfb, dfc,dfd]
    if dbx:
        df=list_db[dbx-1]
        
    else:
        df = pandas.concat(list_db, ignore_index=True).reset_index()
    pandas.to_numeric(df["Pcdays"])
    
    df=df.sort_values(by=["Pcdays"])
    
    df = df.filter(items=["Value", 'Species', 'Parameter', "Pcdays",'%', "Repeated",'DOIs','author'])
    #looking for duplicates throw different authors and taking the mean value
    # df.loc['Rat'].groupby('Parameter').mean().reset_index()
    # df.loc['Human'].groupby('Parameter').mean().reset_index()
    # mask1 =df['Parameter'].duplicated(subset=['Human'],keep=False)
    # df.loc[mask1, 'Rat'] += bigpict.groupby('Human').cumcount().add(0.1)
    return df


def ranged(df,a,b, graph=False,rodent='Rat'):
    # returns a numpy matrix with Start and End points and optionaly a graph

    df2 = df.copy()
    df2 = df2.loc[(df2["Value"] == 'Start') | (df2["Value"] == 'End')]
    
    group = df2.pivot(index=['Parameter', "Value"],
                      columns='Species', values="Pcdays")#.dropna()
    
    group.reset_index(level=None, inplace=True)
    
    if graph:
        milestones =np.array([[1,1],[280,22]])
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
    
    # # duplicates=puntual['Parameter'][puntual.Species == 'Human'].duplicated(keep=False)
    # print(puntual.shape)
    # print(puntual[puntual.Species == 'Human'].groupby('Parameter')['Pcdays'].mean().reset_index().head())
    # #.transform('mean')
    # #puntual[puntual.Species == 'Human'].groupby('Parameter')['Pcdays'].transform('mean').reset_index()
    # #puntual[puntual.Species == 'Human']=puntual[puntual.Species == 'Human'].groupby('Parameter')['Pcdays'].mean()
    # puntual_H=puntual[puntual.Species == 'Human'].groupby('Parameter')['Pcdays'].mean().reset_index()
    # puntual_R=puntual[puntual.Species == 'Rat'].groupby('Parameter')['Pcdays'].mean().reset_index()
    # puntual = pandas.concat([puntual_R,puntual_H])
    # print(puntual[puntual.Species == 'Rat'].groupby('Parameter')['Pcdays'].mean().reset_index().head())
    # print(puntual.shape)
    # #print(puntual.groupby('Parameter').mean().reset_index())
    
    puntual2 = puntual[['Parameter', 'Species', "Pcdays"]].groupby(
        ['Parameter', 'Species'], axis=0, as_index=True).aggregate('first').unstack()
    
    
    
    Pcdays = puntual2['Pcdays'].copy()
    Pcdays = Pcdays.dropna(subset=['Human', rodent])
    # print(Pcdays.shape)
    
    # #Pcdays = Pcdays.groupby('Parameter').mean().reset_index()
    # grupos_filtrados =  Pcdays.groupby('Parameter').filter(lambda x: len(x) == 1)
    # Pcdays['Human'] = pandas.to_numeric(Pcdays['Human'], downcast="float")
    # print(Pcdays.head())
    # print(Pcdays.shape)
    # print(grupos_filtrados)
    # Pcdays[rodent] = pandas.to_numeric(Pcdays[rodent], downcast="float")
    
    
    
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

#df=reader()# 0=, 1=campos-iorii 2= Godlewski 3=Clancy 4= Ohmura

campos_iorii=reader(dbx=1)
Godlewski=reader(dbx=2)
Clancy=reader(dbx=3)
Ohm=reader(dbx=4)
#ohm_cla_godl=pandas.concat([reader(dbx=2),reader(dbx=3),reader(dbx=4)], ignore_index=True).reset_index()
All=reader()

#np.savetxt("file1ccvcv.txt", arraybw)
#graph_ohm_cla_godl = puntual(ohm_cla_godl, ranges=True ,rodent='Rat')#,graph=False, colors =palette[0]

graph_All = puntual(All, ranges=True ,rodent='Rat')#,graph=False, colors =palette[5]
params = logreg1(graph_All)
with open('databases.txt', 'a') as file:
    file.write(f'campos_iorii rows ={campos_iorii.shape[0]}' )
    file.write(f'Godlewski rows ={Godlewski.shape[0]}'  )
    file.write(f'Clancy rows ={  Clancy.shape[0]}'  )
    file.write(f'Ohm rows ={ Ohm.shape[0] }' )
    file.write(f'ALL rows ={ All.shape[0] }' )
    file.write(f'a,b ={ params }' )
print (All.shape)
#%%
#figure 1
figure1=plt.figure(1,constrained_layout=True)#,layout="constrained",figsize=(11.69,8.27)

milestones = [[1, 280], [1, 22]]
graph_CI = puntual(campos_iorii, ranges=True ,rodent='Rat')#
graph_Godlewski= puntual(Godlewski, ranges=True ,rodent='Rat')#graph=True,row=1, colors =palette[2]
graph_clancy= puntual(Clancy, ranges=True ,rodent='Rat')#graph=True,row=2, colors =palette[3],
graph_Ohm= puntual(Ohm, ranges=True ,rodent='Rat')#graph=True, row=3,colors =palette[4],

spec1 = figure1.add_gridspec(5, 3)
#plt.rcParams['figure.constrained_layout.use'] = True
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
graph_Ohm_pd = puntual(Ohm, ranges=True ,pandas_df=True)

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
#plt.savefig('fig1.png')
#%%
########################################suplementary1 (fig 2 and 3)

ols(graph_CI,params[0],params[1],'graph_CI',colors =palette[1])
ols(graph_Godlewski,params[0],params[1],'graph_Godlewski',colors =palette[2])
ols(graph_clancy,params[0],params[1],'graph_clancy',colors =palette[3])
ols(graph_Ohm,params[0],params[1],'graph_Ohm',colors =palette[4])
ols(graph_All,params[0],params[1],'graph_All',colors =palette[5])

nlrsm(graph_CI,'graph_CI',params[0],params[1])
nlrsm(graph_Godlewski,'graph_Godlewski',params[0],params[1])
nlrsm(graph_clancy,'graph_clancy',params[0],params[1])
nlrsm(graph_Ohm,'graph_Ohm',params[0],params[1])
nlrsm(graph_All,'graph_All',params[0],params[1])

#%%
#figure 2,all 4 data bases together 
milestones = [[1, 280], [1, 22]]

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
#plt.savefig('fig2.png')
figure2.savefig('fig2.tif', format='tif', dpi=450)

#%%
#FIGURE 3
figure3=plt.figure(6,figsize=(11.69,8.27))#return slope, intercept,x,y, y_pred,backlog, backlogy

spec3 = figure3.add_gridspec(3, 3)

tuple_qr_bottom =QLRegression_subplot(graph_All, 0.05,params[0],params[1])# returns slope, intercept,x,y, y_pred,backlog, backlogy
tuple_qr_median =QLRegression_subplot(graph_All, 0.5,params[0],params[1])
tuple_qr_top =QLRegression_subplot(graph_All, 0.95,params[0],params[1])    

table1=np.stack((tuple_qr_bottom[-2], tuple_qr_bottom[-1],tuple_qr_top[-1]), axis=1).reshape(-1,3)
table1 = pandas.DataFrame(table1, columns = ['DAF Human','DAF Rat bottom','DAF Rat top'])#,'DAF Rat Top'
table1.to_excel('table1.xlsx', sheet_name='new_sheet_name')

figure3a=figure3 .add_subplot(spec3[0,:])
plt.scatter(tuple_qr_bottom[2],tuple_qr_bottom[3],c='#2CBDFE',alpha=0.5)
plt.plot(tuple_qr_bottom[2], tuple_qr_bottom[4],c='y', label="Quantile: 0.05")
plt.plot(tuple_qr_median[2], tuple_qr_median[4],c='g', label="Quantile: 0.5")
plt.plot(tuple_qr_top[2], tuple_qr_top[4],c='b', label="Quantile: 0.95")
milestones =np.array([[1,1],[280,22]])

plt.scatter(func1(milestones[:,0],params[0],params[1]),milestones [:,1],c='r') 
figure3a.set_title('a', fontsize='small', loc='left')
plt.xlabel("Human' ")
plt.ylabel("Rat (days after fertilization)")
plt.legend(loc='upper left')
plt.xlim(0)
plt.ylim(0)

figure3b=figure3 .add_subplot(spec3[1,:])
plt.scatter(graph_All[:,0],graph_All[:,1],c='#2CBDFE',alpha=0.5)
plt.plot(tuple_qr_bottom[-2], tuple_qr_bottom[-1],c='y', label="Quantile: 0.05")#backlog, backlogy,label=f"Quantile: {quantile}")
plt.plot(tuple_qr_median[-2], tuple_qr_median[-1],c='g', label="Quantile: 0.5")
plt.plot(tuple_qr_top[-2], tuple_qr_top[-1],c='b', label="Quantile: 0.95")
milestones =np.array([[1,1],[280,22]])

plt.scatter(milestones[:,0],milestones [:,1],c='r') 

figure3b.set_title('b', fontsize='small', loc='left')
plt.xlabel("Human (days after fertilization)")
plt.ylabel("Rat (days after fertilization)")
plt.legend(loc='upper left')
plt.xlim(0)
plt.ylim(0)
figure3c=figure3 .add_subplot(spec3[2,0])#.margins(x=0, y=-0.25)
plt.xlim(0, 280)
plt.ylim(0, 30) 
plt.scatter(graph_All[:,0],graph_All[:,1],c='#2CBDFE',alpha=0.5)
plt.plot(tuple_qr_bottom[-2], tuple_qr_bottom[-1],c='y')#backlog, backlogy,label=f"Quantile: {quantile}")
plt.plot(tuple_qr_median[-2], tuple_qr_median[-1],c='g')
plt.plot(tuple_qr_top[-2], tuple_qr_top[-1],c='b')
milestones =np.array([[1,1],[280,22]])
figure3c.set_title('c', fontsize='small', loc='left')
plt.scatter(milestones[:,0],milestones [:,1],c='r') 
figure3d=figure3 .add_subplot(spec3[2,1])#.margins(x=0, y=-0.25) 
plt.scatter(graph_All[:,0],graph_All[:,1],c='#2CBDFE',alpha=0.5)
plt.plot(tuple_qr_bottom[-2], tuple_qr_bottom[-1],c='y')#backlog, backlogy,label=f"Quantile: {quantile}")
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
plt.plot(x_fig3, func1(x_fig3,params[0],params[1]),c='#2CBDFE',label="Log fit")
plt.scatter(milestones[:,0],milestones [:,1],c='r') 
figure3e.set_title('e', fontsize='small', loc='left')
plt.xlim(0)
plt.ylim(0)
plt.legend(loc='upper left')
#plt.savefig('fig3.png')
figure3.savefig('fig3.tif', format='tif', dpi=450)
#%%
#figure 4



ranged(campos_iorii,params[0],params[1],graph=True)
#%%

################################FIGURE 5
 ###############################'Brain Weight'
figure5=plt.figure(9,figsize=(11.69,8.27))
spec5 = figure5.add_gridspec(4, 2)
dfc_H,dfc_H2, dfc_R,dfc_R2=contiguous1(campos_iorii )#,graph_All
figure5a=figure5 .add_subplot(spec5[0,0])


ax1=sns.lineplot(data=dfc_H , x="Pcdays", y='%',hue = "Repeated",linestyle='--',legend=False)
ax1.set(xlabel='Human (DAF)', ylabel='% of adult \n brain weight')
plt.plot(dfc_H2[:,0],dfc_H2[:,1])
plt.xlim(0,3000)
plt.axvline(milestones[1][0], color='r')
figure5a.set_title('a', fontsize='small', loc='left')
figure5b=figure5 .add_subplot(spec5[0,1])
ax2=sns.lineplot(data=dfc_R,x="Pcdays", y='%',hue="Repeated",linestyle='--',legend=False)
ax2.set(xlabel='Rat (DAF)',ylabel=None)
plt.plot(dfc_R2[:,0],dfc_R2[:,1])
plt.xlim(0,150)
plt.axvline(milestones[1][1], color='r')
figure5b.set_title('b', fontsize='small', loc='left')
figure5c=figure5 .add_subplot(spec5[1,0])
interfunction = interp1d( dfc_R2[:,1],dfc_R2[:,0],bounds_error=False, assume_sorted =False)#
arraybw = np.column_stack((dfc_H2, interfunction(dfc_H2[:,1])))

plt.plot(arraybw [:,0],arraybw [:,2])
a=10
for i in arraybw :
    if i[1]>a:
        plt.plot(i[0],i[2],'o' ,color='cyan', alpha=0.5 )
        plt.text(i[0],i[2],str(a)+'%')
        a=a+20

figure5c.set_title('c', fontsize='small', loc='left')
plt.xlim(0, 2000)
plt.ylim(0, 110) 
plt.xlabel("Human (DAF)")
plt.ylabel("Rat (DAF)")
plt.legend(loc='upper left')
plt.axvline(milestones[1][0], color='r')
plt.axhline(milestones[1][1], color='r') 





figure5d=figure5 .add_subplot(spec5[1,1])
plt.scatter(graph_All[:,0],graph_All[:,1],c='#2CBDFE',alpha=0.5)
plt.plot(arraybw [:,0],arraybw [:,2])

plt.xlabel("Human (DAF)")
plt.ylabel("Rat (DAF)")
figure5d.set_title('d', fontsize='small', loc='left')
plt.xlim(0, 1100)
plt.ylim(0,80)
plt.xlabel("Human (DAF)")
plt.ylabel("Rat (DAF)")

print ('////////////////////')



plt.plot(tuple_qr_bottom[-2], tuple_qr_bottom[-1],c='y', label="Quantile: 0.05")#backlog, backlogy,label=f"Quantile: {quantile}")
plt.plot(tuple_qr_median[-2], tuple_qr_median[-1],c='g', label="Quantile: 0.5")
plt.plot(tuple_qr_top[-2], tuple_qr_top[-1],c='b', label="Quantile: 0.95")
milestones =np.array([[1,1],[280,22]])

plt.scatter(milestones[:,0],milestones [:,1],c='r')



figure5e=figure5.add_subplot(spec5[2,0])
GAD_H_Cor,GAD_H_Cor2, GAD_R_Cor,GAD_R_Cor2=contiguous1(campos_iorii, parameter ='GAD activity in cortex' )

ax1=sns.lineplot(data=GAD_H_Cor, x="Pcdays", y='%',hue = "Repeated",linestyle='--',legend=False)
ax1.set(xlabel='Human (DAF)', ylabel='% of adult \n GAD activity in cortex')
plt.plot(GAD_H_Cor2[:,0],GAD_H_Cor2[:,1])

plt.axvline(milestones[1][0], color='r')
figure5e.set_title('e', fontsize='small', loc='left')

figure5f=figure5.add_subplot(spec5[2,1])

ax2=sns.lineplot(data=GAD_R_Cor,x="Pcdays", y='%',hue="Repeated",linestyle='--',legend=False)
ax2.set(xlabel='Rat (DAF)',ylabel=None)
plt.plot(GAD_R_Cor2[:,0],GAD_R_Cor2[:,1])
plt.xlim(0,100)
plt.axvline(milestones[1][1], color='r')
figure5f.set_title('f', fontsize='small', loc='left')
figure5g=figure5.add_subplot(spec5[3,0])

interfunction_GAD_Cor = interp1d( GAD_R_Cor2[:,1],GAD_R_Cor2[:,0],bounds_error=False, assume_sorted =False)
array_GAD_Cor = np.column_stack((GAD_H_Cor2, interfunction_GAD_Cor(GAD_H_Cor2[:,1])))

plt.plot(array_GAD_Cor [:,0],array_GAD_Cor [:,2])

#plt.xlim(0, 2000)
plt.ylim(0, 100) 
plt.xlabel("Human (DAF)")
plt.ylabel("Rat (DAF)")
plt.legend(loc='upper left')
plt.axvline(milestones[1][0], color='r')
plt.axhline(milestones[1][1], color='r') 
figure5g.set_title('g', fontsize='small', loc='left')
figure5h=figure5.add_subplot(spec5[3,1])
plt.scatter(graph_All[:,0],graph_All[:,1],c='#2CBDFE',alpha=0.5)
plt.plot(array_GAD_Cor [:,0],array_GAD_Cor[:,2])
# plt.xlim(0, 800)
# plt.ylim(0,60)
figure5h.set_title('h', fontsize='small', loc='left')
plt.xlabel("Human (DAF)")
plt.ylabel("Rat (DAF)")
#figure6 .add_subplot(spec6[2,1])
# plt.plot(arraybw [:,0],arraybw [:,2])
# # plt.xlim(0, 800)
# # plt.ylim(0,60)
# plt.xlabel("Human (DAF)")
# plt.ylabel("Rat (DAF)")


print ('*/*/*/*/*/')



plt.plot(tuple_qr_bottom[-2], tuple_qr_bottom[-1],c='y', label="Quantile: 0.05")#backlog, backlogy,label=f"Quantile: {quantile}")
plt.plot(tuple_qr_median[-2], tuple_qr_median[-1],c='g', label="Quantile: 0.5")
plt.plot(tuple_qr_top[-2], tuple_qr_top[-1],c='b', label="Quantile: 0.95")
milestones =np.array([[1,1],[280,22]])

plt.scatter(milestones[:,0],milestones [:,1],c='r') 

plt.axvline(milestones[1][0], color='r')

plt.tight_layout()
figure5.savefig('fig5.tif', format='tif', dpi=450)
#%%
#FIGURE suplement 3
figure6=plt.figure(10)
GAD_H_Cer,GAD_H_Cer2, GAD_R_Cer,GAD_R_Cer2=contiguous1(campos_iorii, parameter ='GAD activity in cerebellum' )
spec6 = figure6.add_gridspec(2, 2)

figure6 .add_subplot(spec6[0,0])


ax1=sns.lineplot(data=GAD_H_Cer, x="Pcdays", y='%',hue = "Repeated",linestyle='--',legend=False)
ax1.set(xlabel='Human (DAF)', ylabel='% of adult \n GAD activity in cerebellum')
plt.plot(GAD_H_Cer2[:,0],GAD_H_Cer2[:,1])
#plt.xlim(0,3000)
plt.axvline(milestones[1][0], color='r')

figure6 .add_subplot(spec6[0,1])
ax2=sns.lineplot(data=GAD_R_Cer,x="Pcdays", y='%',hue="Repeated",linestyle='--',legend=False)
ax2.set(xlabel='Rat (DAF)',ylabel=None)
plt.plot(GAD_R_Cer2[:,0],GAD_R_Cer2[:,1])
plt.xlim(0,100)
plt.axvline(milestones[1][1], color='r')

figure6 .add_subplot(spec6[1,0])


interfunction = interp1d( GAD_R_Cer2[:,1],GAD_R_Cer2[:,0],bounds_error=False, assume_sorted =False)
array_GAD_Cer = np.column_stack((GAD_H_Cer2, interfunction(GAD_H_Cer2[:,1])))


plt.plot(array_GAD_Cer [:,0],array_GAD_Cer [:,2])
#
#plt.xlim(0, 2000)
plt.ylim(0, 100) 
plt.xlabel("Human (DAF)")
plt.ylabel("Rat (DAF)")
plt.legend(loc='upper left')
plt.axvline(milestones[1][0], color='r')
plt.axhline(milestones[1][1], color='r') 
#figure5 .add_subplot(spec5[2,0])

#print(data_puntual)
#model=sim(min(data_puntual[:,0]), max(data_puntual[:,0]))
#interfunction = interp1d(array[:,0],array[:,2],bounds_error=False, assume_sorted =False)
# y_pred = interfunction(data_puntual[:,0])
# array2= np.stack((data_puntual[:,1], y_pred.T) , axis=1)
# #print(array2)
# indexList = [np.any(i) for i in np.isnan(array2)]
# #print(indexList)
# array2 = np.delete(array2, indexList, axis=0)
# R_squared=r2_score(array2[:,0], array2[:,1])
# print('R-squared of'+ 'key'+ ' is'+str(R_squared))
# #print('R-squared of the model in this interval  is'+str(R_squared2))
figure6 .add_subplot(spec6[1,1])
plt.scatter(graph_All[:,0],graph_All[:,1],c='#2CBDFE',alpha=0.5)
plt.plot(array_GAD_Cer [:,0],array_GAD_Cer[:,2])
# plt.xlim(0, 800)
# plt.ylim(0,60)
plt.xlabel("Human (DAF)")
plt.ylabel("Rat (DAF)")
#figure6 .add_subplot(spec6[2,1])
# plt.plot(arraybw [:,0],arraybw [:,2])
# # plt.xlim(0, 800)
# # plt.ylim(0,60)
# plt.xlabel("Human (DAF)")
# plt.ylabel("Rat (DAF)")


print ('*/*/*/*/*/')



plt.plot(tuple_qr_bottom[-2], tuple_qr_bottom[-1],c='y', label="Quantile: 0.05")#backlog, backlogy,label=f"Quantile: {quantile}")
plt.plot(tuple_qr_median[-2], tuple_qr_median[-1],c='g', label="Quantile: 0.5")
plt.plot(tuple_qr_top[-2], tuple_qr_top[-1],c='b', label="Quantile: 0.95")
milestones =np.array([[1,1],[280,22]])

plt.scatter(milestones[:,0],milestones [:,1],c='r')
figure6.savefig('fig supl 3.tif', format='tif', dpi=450)


figure7=plt.figure(11,figsize=(100,50))
# TODO # FIXME 
spec7 = figure7.add_gridspec(3)
# campos_iorii_log=campos_iorii.copy()
# campos_iorii_log_R=campos_iorii_log.loc[campos_iorii_log['Species'] == 'Rat']
# campos_iorii_log_R["Pcdays"] = np.log(campos_iorii_log["Pcdays"])

# campos_iorii_log_H=campos_iorii_log.loc[campos_iorii_log['Species'] == 'Human']
# campos_iorii_log_H["s"] = func1(campos_iorii_log["Pcdays"],params[0],params[1])
# campos_iorii_log['Pcdays'][campos_iorii_log.Species== 'Rat'] =np.log( 
#     campos_iorii_log['Pcdays'][campos_iorii_log.Species== 'Rat'])
# campos_iorii_log['Pcdays'][campos_iorii_log.Species== 'Human'] =func1( 
#     campos_iorii_log['Pcdays'][campos_iorii_log.Species== 'Human'],params[0],params[1])





figure7 .add_subplot(spec7[0])
dfc_H,dfc_H2, dfc_R,dfc_R2=contiguous1(campos_iorii )#,graph_All


interfunction = interp1d( dfc_R2[:,1],dfc_R2[:,0],bounds_error=False, assume_sorted =False)
#
arraybw = np.column_stack((dfc_H2, interfunction(dfc_H2[:,1])))
arraybw=arraybw[~(np.isnan(arraybw[:,2]))]
arraybw_X=func1(arraybw [:,0],params[0],params[1])
arraybw_Y=np.log(arraybw [:,2])
interfunction2 = interp1d( arraybw[:,0],arraybw[:,2],bounds_error=False, assume_sorted =False)#

sliced_array_graph_All=graph_All[~(graph_All[:,0]<min(arraybw[:,0]))]
sliced_array_graph_All=sliced_array_graph_All[~(sliced_array_graph_All[:,0]>800)]
contiguos_to_sacatter=interfunction2(sliced_array_graph_All[:,0])


#plt.scatter(func1(sliced_array_graph_All[:,0],params[0],params[1]),np.log(contiguos_to_sacatter),c='r',alpha=0.5)

plt.scatter(func1(sliced_array_graph_All[:,0],params[0],params[1]),np.log(sliced_array_graph_All[:,1]),c='#2CBDFE',alpha=0.5)
x_fig4 = np.linspace(min(sliced_array_graph_All[:,0]), max(sliced_array_graph_All[:,0]),100000)


plt.xlim(20, 45)
R_squared=r2_score(np.log(sliced_array_graph_All[:,1]), np.log(contiguos_to_sacatter))# y true y pred
#print('R-squared of'+ 'lof func'+ ' is'+str(R_squared))
R_squared2=r2_score(np.log(sliced_array_graph_All[:,1]), np.log(func1(sliced_array_graph_All[:,0],params[0],params[1])))# y true y pred
#print('R-squared of'+ 'lof func'+ ' is'+str(R_squared2))
plt.plot(arraybw_X,arraybw_Y,label='BW R-squared =' +str(R_squared))
plt.plot(func1(x_fig4,params[0],params[1]), np.log(func1(x_fig4,params[0],params[1])),c='#2CBDFE',label='log fit R-squared =' +str(R_squared2))
# plt.ylim(0,60)
plt.xlabel("Human'")
plt.ylabel("Rat log (DAF)")
plt.legend(loc='upper left')
# plt.axvline(milestones[1][0], color='r')
# plt.axhline(milestones[1][1], color='r') 
plt.savefig('fig supl 8.png')

#%%
#################fig 7
figure8=plt.figure(12,figsize=(100,50))
#fig7, ax7 = plt.subplots()
bigpict = puntual(All,pandas_df=True, ranges=True ,rodent='Rat')
mask2 =bigpict.duplicated(subset=['Human','Rat'],keep=False)
ax= sns.scatterplot( data=bigpict, x=bigpict['Human'], y=bigpict['Rat'], alpha=0.5)
#sys.exit()

bigpict.loc[mask2, 'Rat'] += bigpict.groupby('Human').cumcount().add(0.1)


for k, v in bigpict.iterrows():
    ax.text(v['Human']+.01, v['Rat'], str(k),fontsize= 1.5,rotation=45)
    
#
plt.ylim(0,60)
plt.savefig('fig supl 4.pdf', dpi=2000, format='pdf')
#plt.savefig('fig supl 4.svg', format='svg', dpi=5000)
#%%
################### confidence intervals 
bootstrapNLR(graph_All,5000,colors =palette[0])
#ax7.annotate(bigpict['Parameter'],bigpict['Human'],bigpict['Rat'])
# #contiguous1(campos_iorii,graph_CI)
# # # df=df.loc[df["Pcdays"] < 1100]
# # # df=df.loc[df["Pcdays"] > 17]
# # contiguous1(df,graph)#df,data_puntual=False,graph=False,rodent='Rat', if data_puntual is provided the function calculates goodnes of fit
# # #contiguous2(df,graph)
#params_CI = logreg1(graph_CI,colors =palette[1])
#params_graph_ohm_cla_godl = logreg1(graph_ohm_cla_godl,colors =palette[0])
#
# #print()
# quant_df = puntual(df, pandas_df=True)
# #graph=np.concatenate(puntual(df), ranged(df,params[0],params[1]))



#$
# #print(graph.shape)
#print(params_graph_ohm_cla_godl)
#QLRegression2(graph_All, 0.5,params[0],params[1])
# for i in [0.05,0.5,0.95]:
#QLRegression2(graph, i,params[0],params[1])

# # rangedcutter(ranged(df))


# QLRegression_df(graph, [0.1  , 0.9],params[0],params[1],quant_df)
# # logreg2(graph)

# graph[np.all(graph != 0, axis=1)]
slopes(graph_All, 0.5,params[0],params[1])
#bootstrapNLR(graph_ohm_cla_godl,5000,colors =palette[0])
#bootstrapNLR(graph_CI,5000,colors =palette[1])
#bootstrapNLR(graph_All,5000,colors =palette[5])
#bootstrap(graph, 0.5,100,params[0],params[1])
#splines(graph, 0.5,280, 300,params[0],params[1])
#bootstrap(graph, 0.5,100,params[0],params[1])

plt.show()