import pandas  as pd #For reading data
import numpy as np #For data analysis
import datetime #For delaing with date-time conversion/formats
from matplotlib import pyplot as plt #For Data visualization

Data = pd.read_csv('/Users/nikhil/Data/ML_examples/NYC_Wi-Fi_Hotspot_Locations.csv')
Columns = Data.columns
#Using numpy
DATA = np.array(Data)
idx_provider = np.where(Columns=='Provider')[0][0]
idx_location = np.where(Columns=='Location')[0][0]
idx_boroname = np.where(Columns=='Borough Name')[0][0]
idx_location_T = np.where(Columns=='Location_T')[0][0]
idx_NTA = np.where(Columns=='Neighborhood Tabulation Area (NTA)')[0][0]
idx_NTA_code = np.where(Columns=='Neighborhood Tabulation Area Code (NTACODE)')[0][0]
idx_type = np.where(Columns=='Type')[0][0]
idx_lat = np.where(Columns=='Latitude')[0][0]
idx_lon = np.where(Columns=='Longitude')[0][0]
idx_activated = np.where(Columns=='Activated')[0][0]
UNIQUE_PROVIDERS = len(np.unique(DATA[:,idx_provider])) #Answer to question 1

BRONX_DATA = DATA[np.where(DATA[:,idx_boroname]=='Bronx')[0],:] #Select hotspots in bronx only
BRONX_PROVIDERS,COUNTS = np.unique(BRONX_DATA[:,idx_provider],return_counts=True) #Counts array is the answer to question 2
PARK_HOTSPOTS = 0
NO_LIBRARY = []
for index in range (0,len(DATA)):
    string = DATA[index,idx_location].split() #for question 3
    string_not_library = DATA[index,idx_location_T].split() #for question 4
    string = [each_string.lower() for each_string in string]
    string_not_library = [each_string_nl.lower() for each_string_nl in string_not_library]
    if 'park' in string: #loop for question 3
        PARK_HOTSPOTS +=1
    if 'library' not in string_not_library: #loop for question 4
        NO_LIBRARY.append(index)
FRACTION_PARK_HOTSPOTS = PARK_HOTSPOTS/len(DATA) #Answer to question 3

#To answer question 4 lets first select those hotspots which are not in the library and call them as NLIB.
NLIB = DATA[np.array(NO_LIBRARY),:] #Length of NLIB array is now our sample space
FREE_WIFI = np.where(NLIB[:,idx_type]=='Free')[0]
PROB_FREE = len(FREE_WIFI)/len(NLIB) #Answer to question 4


#Calculation for question 5
NTA_CODE,NTA_COUNT = np.unique(DATA[:,idx_NTA_code],return_counts=True) #search by NTA code rather than NTA name
NTA_VALID,NTA_VALID_COUNTS = NTA_CODE[NTA_COUNT>30],NTA_COUNT[NTA_COUNT>30]
NTA_Data = pd.read_csv('/Users/nikhil/Data/ML_examples/Census_Demographics_at_the_Neighborhood_Tabulation_Area__NTA__level.csv')
NTA_Columns = NTA_Data.columns
NTA_DATA = np.array([NTA_Data])[0,:,:]
idxnta_nta_code = np.where(NTA_Columns=='Geographic Area - Neighborhood Tabulation Area (NTA)* Code')[0][0]
idxnta_population = np.where(NTA_Columns=='Total Population 2010 Number')[0][0]
NTA_POPULATION = np.zeros(len(NTA_VALID))
for j in range (0,len(NTA_VALID)):
    cross_index = np.where(NTA_DATA[:,idxnta_nta_code]==NTA_VALID[j])[0][0]
    NTA_POPULATION[j] = NTA_DATA[cross_index,idxnta_population]
NUM_HOTSPOTS_PERCAPITA = NTA_POPULATION/NTA_VALID_COUNTS
IQR = np.percentile(NUM_HOTSPOTS_PERCAPITA,75) - np.percentile(NUM_HOTSPOTS_PERCAPITA,25)

#Calculation of distance travelled between hotspots
AVG_DISTANCE = np.zeros((len(DATA)))
LAT = (DATA[:,idx_lat].astype(float))*0.0174533 #convert to radians
LON = (DATA[:,idx_lon].astype(float))*0.0174533
Radius = 6371
for d in range (0,len(DATA)):
    d_phi = LAT[d] - LAT
    d_lambda = LON[d] - LON
    phi_m = (LAT[d] + LAT)/2
    D = Radius*np.sqrt((d_phi)**2 + (np.cos(phi_m)*d_lambda)**2)
    AVG_DISTANCE[d] = np.mean(D[D.argsort()][1:4])*3280 #convert to feet
DIST_BETWEEN_HOTSPOTS = np.median(AVG_DISTANCE) #Answer to question 6


#Calculation for question 7
DATE_ACTIVATED = DATA[:,idx_activated]
DAY_ACTIVATED = np.zeros(len(DATE_ACTIVATED),dtype=int)
MONTH_ACTIVATED = np.zeros(len(DATE_ACTIVATED),dtype=int)
YEAR_ACTIVATED = np.zeros(len(DATE_ACTIVATED),dtype=int)
DAY_ACTIVATED_NAME = np.empty(len(DATE_ACTIVATED),dtype=object)
for w in range (0,len(DATE_ACTIVATED)):
    SPLIT_DATE = DATE_ACTIVATED[w].split('/')
    MONTH_ACTIVATED[w] = int(SPLIT_DATE[-3])
    DAY_ACTIVATED[w] = int(SPLIT_DATE[-2])
    YEAR_ACTIVATED[w] = int(SPLIT_DATE[-1])
    NAME = datetime.date(int(SPLIT_DATE[-1]),int(SPLIT_DATE[-3]),int(SPLIT_DATE[-2]))
    DAY_ACTIVATED_NAME[w] = NAME.strftime("%A")
BAD_YEAR = YEAR_ACTIVATED == 9999
GOOD_DAY = DAY_ACTIVATED[~BAD_YEAR]
GOOD_MONTH = MONTH_ACTIVATED[~BAD_YEAR]
GOOD_YEAR = YEAR_ACTIVATED[~BAD_YEAR]
GOOD_DAY_NAME = DAY_ACTIVATED_NAME[~BAD_YEAR]
WEEK,COUNTS_DAY = np.unique(GOOD_DAY_NAME,return_counts=True)
DAY_MOST_ACTIVATIONS,MOST_ACTIVATIONS = WEEK[np.argmax(COUNTS_DAY)],COUNTS_DAY[np.argmax(COUNTS_DAY)] #The Day when most activations occurred, Number of Activation on that day
ALL_VALID_ACTIVATIONS = len(GOOD_DAY) #Only considering those cases with valid dates
FRACTION_MOST_ACTIVATIONS = MOST_ACTIVATIONS/ALL_VALID_ACTIVATIONS #Answer to question 7


#Calculation to question 8
start_date = datetime.datetime(np.min(GOOD_YEAR),np.min(GOOD_MONTH),np.min(GOOD_DAY))
cut_date = datetime.datetime(2018,7,1)
NUM_MONTHS = np.zeros(len(GOOD_MONTH))
for N in range (0,len(GOOD_MONTH)):
    end_date = datetime.datetime(GOOD_YEAR[N],GOOD_MONTH[N],GOOD_DAY[N])
    NUM_MONTHS[N] = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
CUT_OFF_MONTH = (cut_date.year - start_date.year) * 12 + (cut_date.month - start_date.month)
CUT_OFF_NUM_MONTHS = NUM_MONTHS<30
X_variable,Y_variable = np.unique(NUM_MONTHS[CUT_OFF_NUM_MONTHS],return_counts=True)
#We will use the exact solution of OLS to get the slope and intercept. First we will plug a column of 1 in X_variable to take care of the constant. This will allow us to directly evaluate the output parameter array theta, that contain b and m such that theta = [b,m]
X_train = np.c_[np.ones((X_variable.shape[0],1)),X_variable]
Y_train = Y_variable
theta = np.matmul(np.linalg.inv(np.matmul(X_train.T,X_train) ),np.matmul(X_train.T,Y_train))#theta[1] is answer to question 8

fig=plt.figure()
f1=20
lw=1.5
ltw = 8
ltwm = 3
pad_space = 5
plt.plot(X_variable,Y_variable,c='r',label='Data',linewidth=lw)
plt.plot(X_variable,theta[1]*X_variable+theta[0],c='k',label='Fit(m = '+str(np.round(theta[1],2))+';b = '+str(np.round(theta[0],2))+')',linewidth=lw)
plt.legend(fontsize=f1-10)
plt.xlabel('N$_{Months}$ since hotspot activation',fontsize=f1)
plt.ylabel('N$_{Activations}$',fontsize=f1)
plt.minorticks_on()
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=f1,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=f1,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=f1,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=f1,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(lw)
fig.set_size_inches(13,6)
