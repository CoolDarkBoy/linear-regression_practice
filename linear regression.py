import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#headbrain.csv was obtained from kaggle
data=pd.read_csv("headbrain.csv")


#print(data.shape)
#print(data.head())

x = data['Head Size(cm^3)'].values
y = data['Brain Weight(grams)'].values

mean_x=np.mean(x)
mean_y=np.mean(y)
a=len(x)

numer=0
denom=0

for i in range(a):
    numer+=(x[i]-mean_x)*(y[i]-mean_y)
    denom+=(x[i]-mean_x)**2
m=numer/denom

#y = m*x-c
#for finding c, c = (m*mean_x)- mean_y
c=mean_y-(m*mean_x)

#finally finding linear regression line
Y=m*x+c

#plotting
plt.plot(x,Y,label='Regression Line',color='c')

#plotting scatter
plt.scatter(x,y,label='Scatter Plot',color='r')
plt.xlabel('Head Size(cm^3)')
plt.ylabel('Brain Weight(grams)')
plt.legend()
plt.show()


#To check goodness of fit
fit_num=0
fit_den=0
for i in range(a):
    #y=mx+c
    y_pred=(m*x[i])+c
    fit_num+=(Y[i]-y_pred)
    fit_den+=(Y[i]-mean_y)
r2=1-(fit_num/fit_den)
print(r2)
