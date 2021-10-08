import numpy as np
from itertools import permutations
from matplotlib import pyplot as plt

def laser_dist(heights,positions):
    L = np.zeros(len(heights))
    L[0] = 1
    for l in range (1,len(L)):
        diff = heights-heights[l]
        if np.all(diff[:l]<0):
            L[l] = positions[l]
        else:
            index_positive = np.where(diff[:l]>=0)[0]
            L[l] = positions[l] - np.max(index_positive) - 1
    return L

def output_stat(H_input):
    x = np.linspace(1,len(H_input),len(H_input))
    Config_all = list(set(permutations(H_input)))
    Dist_config = np.zeros(len(Config_all))
    for c in range (0,len(Config_all)):
        heights = np.array(Config_all[c])
        Dist_config[c]=sum(laser_dist(heights,x))
    mean = np.sum(Dist_config)/len(Dist_config)
    std = np.sqrt(np.sum((Dist_config-mean)**2)/len(Dist_config))
    return mean,std


H_input_1 = np.array([1,1,3,4]) #For H=[1,1,3,4]
mean_1,std_1 = output_stat(H_input_1)


H_input_2 = np.linspace(1,10,10) #For H=[1,2,....N] when N=10
mean_2,std_2 = output_stat(H_input_2)



#for N = 20, there are a lot of configurations. We can establish a relation between mean and N for different values of N.
N = np.array([4,5,6,7,8,9,10,11])
mean = np.zeros(len(N))
std = np.zeros(len(N))
for n in range (0,len(N)):
    H_input = np.linspace(1,N[n],N[n])
    mean[n],std[n] = output_stat(H_input)

X_variable,Y_variable = N,mean

#try a 2nd order polynomial fit
X_train = np.c_[np.ones((X_variable.shape[0],1)),X_variable,X_variable**2,X_variable**3]
Y_train = Y_variable
theta = np.matmul(np.linalg.inv(np.matmul(X_train.T,X_train) ),np.matmul(X_train.T,Y_train))
Pred_Y = theta[0] + X_variable*theta[1]+ (X_variable**2)*theta[2]+ (X_variable**3)*theta[3]
mse = 1/len(Y_train)*np.sum((Y_train-Pred_Y)**2)

fig=plt.figure()
f1=20
lw=1.5
ltw = 8
ltwm = 3
pad_space = 5
plt.plot(X_variable,Y_variable,c='r',label='Data',linewidth=lw)
plt.plot(X_variable,Pred_Y,c='k',label='Fit',linewidth=lw)
plt.legend(fontsize=f1-10)
plt.xlabel('N',fontsize=f1)
plt.ylabel('mean',fontsize=f1)
plt.minorticks_on()
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=f1,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=f1,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=f1,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=f1,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(lw)
fig.set_size_inches(13,6)


#For N=20
X_new=20
mean_N20 = theta[0] + X_new*theta[1]+ (X_new**2)*theta[2]+ (X_new**3)*theta[3]
