import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
import corner


def StandardScaler(Data):
    for feature in range (0,Data.shape[1]):
        Current_feature = Data.iloc[:,feature]
        feature_std = (Current_feature - np.mean(Data.iloc[:,feature]))/(np.std(Data.iloc[:,feature]))
        Data.iloc[:,feature].update(feature_std)
    return Data

def MinMaxScaler(Data,amax,amin):
    for feature in range (1,Data.shape[1]):
        Current_feature = Data.iloc[:,feature]
        feature_std = (Current_feature - Data.iloc[:,feature].min())/(Data.iloc[:,feature].max()-Data.iloc[:,feature].min())
        feature_normalized = (feature_std*(amax-amin))+amin
        Data.iloc[:,feature].update(feature_normalized)
    return Data

Data = pd.read_csv('/Users/nikhil/Data/ML_examples/airfoil_self_noise.dat',sep='\t',header=None)
Data.columns = ['F','theta','L','v','T','P']

#Lets perform some Exploratory Data Analysis

#First Check for NaNs or missing values
print(np.sum(Data.isnull()))

#Outlier removal
DATA = np.array(Data)
DATA_Q = DATA
for cols in range (0,DATA.shape[1]):
    Q1 = np.quantile(DATA_Q[:,cols],0.25)
    Q3 = np.quantile(DATA_Q[:,cols],0.75)
    IQR = Q3 - Q1
    low_cut = Q1 - (1.5 * IQR)
    high_cut = Q3 + (1.5 * IQR)
    good_data = np.where((DATA_Q[:,cols]>=low_cut)&(DATA_Q[:,cols]<=high_cut))[0]
    DATA_Q = DATA_Q[good_data,:]

#Adding simulated noise to the data
Least_count_instruments = np.array([50,0.5,0,0,0.0001,1])
NOISY_DATA_Q = np.zeros(np.shape(DATA_Q))
for cols in range (0,DATA_Q.shape[1]):
    for rows in range (0,DATA_Q.shape[0]):
        NOISY_DATA_Q[rows,cols] = DATA_Q[rows,cols] + (np.random.normal(0,1)*Least_count_instruments[cols])
        
NOISY_DATA_Q[:,1] = np.clip(NOISY_DATA_Q[:,1],0,None)




ft=20
lw=1.5
ltw=8
ltwm=3
pad_space = 5

#Plotting boxplots for original data
fig=plt.figure()
plt.subplot(2,3,1)
plt.boxplot(DATA[:,0])
plt.ylabel('F (Hertz)',fontsize=ft)
plt.minorticks_on()
plt.xticks([])
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=0,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=0,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(lw)

plt.subplot(2,3,2)
plt.boxplot(DATA[:,1])
plt.ylabel('theta (degrees)',fontsize=ft)
plt.minorticks_on()
plt.xticks([])
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=0,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=0,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(lw)

plt.subplot(2,3,3)
plt.boxplot(DATA[:,2])
plt.ylabel('L (meters)',fontsize=ft)
plt.minorticks_on()
plt.xticks([])
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=0,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=0,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(lw)

plt.subplot(2,3,4)
plt.boxplot(DATA[:,3])
plt.ylabel('v (meters/sec)',fontsize=ft)
plt.minorticks_on()
plt.xticks([])
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=0,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=0,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(lw)

plt.subplot(2,3,5)
plt.boxplot(DATA[:,4])
plt.ylabel('T (meters)',fontsize=ft)
plt.minorticks_on()
plt.xticks([])
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=0,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=0,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(lw)

plt.subplot(2,3,6)
plt.boxplot(DATA[:,5])
plt.ylabel('P (decibels)',fontsize=ft)
plt.minorticks_on()
plt.xticks([])
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=0,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=0,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(lw)

fig.subplots_adjust(left=0.093, bottom=0.032, right=0.99, top=0.977, wspace=0.393, hspace=0.087)
fig.set_size_inches(20,7)


#Making corner plot
labels = ['F(Hertz)','theta(degrees)','L(meters)','v(meters/sec)','T(meters)','P(decibels)']
fig_corner = corner.corner(DATA_Q,plot_datapoints=True,labels=labels,label_kwargs=dict(fontsize=12),color='r',s=20,plot_density=False,plot_contours=True)
fig_corner.set_size_inches(10,7.5)



#Making pair plot with nosiy and original data
fig1=plt.figure()
plt.subplot(1,3,1)
plt.scatter(DATA_Q[:,0],DATA_Q[:,-1],s=10,c='k',label='Original Data')
plt.scatter(NOISY_DATA_Q[:,0],NOISY_DATA_Q[:,-1],s=20,facecolor='none',edgecolor='r',label='Nosiy Data')
plt.xlabel('F (Hertz)',fontsize=ft)
plt.ylabel('P (decibels)',fontsize=ft)
plt.minorticks_on()
plt.legend(fontsize=ft-10)
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft,pad=pad_space,axis='both',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft,pad=pad_space,axis='both',bottom=True,top=True,left=True,right=True)
for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(lw)


plt.subplot(1,3,2)
plt.scatter(DATA_Q[:,1],DATA_Q[:,-1],s=10,c='k',label='Original Data')
plt.scatter(NOISY_DATA_Q[:,1],NOISY_DATA_Q[:,-1],s=20,facecolor='none',edgecolor='r',label='Nosiy Data')
plt.xlabel('theta (Hertz)',fontsize=ft)
#plt.ylabel('P (decibels)',fontsize=ft)
plt.minorticks_on()
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=0,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=0,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(lw)

plt.subplot(1,3,3)
plt.scatter(DATA_Q[:,4],DATA_Q[:,-1],s=10,c='k',label='Original Data')
plt.scatter(NOISY_DATA_Q[:,4],NOISY_DATA_Q[:,-1],s=20,facecolor='none',edgecolor='r',label='Nosiy Data')
plt.xlabel('T (meters)',fontsize=ft)
#plt.ylabel('P (decibels)',fontsize=ft)
plt.minorticks_on()
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=0,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=0,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(lw)

fig1.subplots_adjust(left=0.08, bottom=0.177, right=0.979, top=0.976, wspace=0.0, hspace=0.2)
fig1.set_size_inches(20,5)
