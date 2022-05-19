"""
    Name: G.K.V.Snigdha
    Roll No.: B20195
    Mobile No.: 9711054392
    Branch: Computer Science Engineering
"""


import math
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg as AR
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = pd.read_csv(r'daily_covid_cases.csv')


#-------------------------------------------------------Question - 1-------------------------------------------------------#
print("#-------------------------------------------------------Question - 1-------------------------------------------------------#")

#---Part(a)---#
print("#---Part(a)---#")
print()
print()

months = ["Feb-20","Apr-20","Jun-20","Aug-20","Oct-20","Dec-20","Feb-21","Apr-21","Jun-21","Aug-21","Oct-21"]

x = [16]
for i in range(10):
    x.append(x[i] + 60)

plt.figure(figsize=(20, 10))
plt.xticks(x, months)
plt.xlabel('Month-Year')
plt.ylabel('New Confirmed Cases')
plt.title('No. of confirmed cases vs Months')
plt.plot(data["new_cases"])
plt.show()


print()
print()



#---Part(b)---#
print("#---Part(b)---#")
print()
print()


lag = [0]

for i in range(len(data)-1):
    lag.append(data["new_cases"][i])

print("The Autocorrelation between the time sequence and its one-day lag is:",round(np.corrcoef(data["new_cases"],lag)[0][1],3))


print()
print()


#---Part(c)---#
print("#---Part(c)---#")
print()
print()


plt.scatter(data["new_cases"],lag)
plt.xlabel('Time Series')
plt.ylabel('Time Series with time-lag 1')
plt.title('No. of confirmed cases vs No. of confirmed cases with time lag 1')
plt.show()

print()
print()


#---Part(d)---#
print("#---Part(d)---#")
print()
print()


lag_values = []

for i in range(1,7,1):
    
    lag = data["new_cases"].shift(i)
    lag_values.append(round(np.corrcoef(data["new_cases"][i:],lag[i:])[0][1],3))
    print("The Autocorrelation between the time sequence and its",i,"lag is:",round(np.corrcoef(data["new_cases"][i:],lag[i:])[0][1],3))

plt.plot([1,2,3,4,5,6],lag_values)
plt.xlabel("Lagged values (p)")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation vs Lagged Values (p)")
plt.show()

print()
print()


#---Part(e)---#
print("#---Part(e)---#")
print()
print()

sm.graphics.tsa.plot_acf(data["new_cases"], lags=6)
plt.show()

print()
print()



#-------------------------------------------------------Question - 2-------------------------------------------------------#
print("#-------------------------------------------------------Question - 2-------------------------------------------------------#")


#---Part(a)---#
print("#---Part(a)---#")
print()
print()


series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]



p = 5

model = AR(train, lags=p, old_names=True)

model_fit = model.fit()

coef = model_fit.params

print(coef)

print()
print()


#---Part(b)---#
print("#---Part(b)---#")
print()
print()



history = train[len(train)-p:]
history = [history[i] for i in range(len(history))]
predictions = list()

for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-p,length)]
    yhat = coef[0]
    for d in range(p):
        yhat += coef[d+1] * lag[p-d-1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)

    
#-------- (i)

plt.scatter(predictions, test)
plt.xlabel("Actual values")
plt.ylabel("Predicted Values")
plt.title("Scatter Plot for Actual vs Predicted values for Time Lag = 5")
plt.show()

#-------- (ii)

plt.figure(figsize=(20,10))
plt.xlabel("Days")
plt.ylabel("Actual values and Predicted Values")
plt.title("Line Plot for Actual vs Predicted values for Time Lag = 5")
plt.plot(predictions)
plt.plot(test)
plt.show()

#-------- (iii)

T = len(test)
s1 = 0
s2 = 0

for t in range(T):
    diff1 = (predictions[t]-test[t])**2
    diff2 = (abs(test[t]-predictions[t]))/test[t]
    s1 += diff1
    s2 += diff2

num = np.sqrt(s1/T)
den = np.sum(test)/T

RMSE = (num/den)*100
MAPE = (s2/T)*100

print("RMSE Error is:",round(RMSE[0],3))
print("MAPE Error is:",round(MAPE[0],3))



print()
print()



#-------------------------------------------------------Question - 3-------------------------------------------------------#
print("#-------------------------------------------------------Question - 3-------------------------------------------------------#")



P = [1,5,10,15,25]
RMSE_Lag = []
MAPE_Lag = []

for p in P:
    
    model = AR(train, lags=p, old_names=True)

    model_fit = model.fit()

    coef = model_fit.params
    
    history = train[len(train)-p:]
    history = [history[i] for i in range(len(history))]
    predictions = list()

    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-p,length)]
        yhat = coef[0]
        for d in range(p):
            yhat += coef[d+1] * lag[p-d-1]
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)
      
    T = len(test)
    s1 = 0
    s2 = 0

    for t in range(T):
        diff1 = (predictions[t]-test[t])**2
        diff2 = (abs(test[t]-predictions[t]))/test[t]
        s1 += diff1
        s2 += diff2

    num = np.sqrt(s1/T)
    den = np.sum(test)/T

    RMSE = (num/den)*100
    MAPE = (s2/T)*100
    
    RMSE_Lag.append(RMSE[0])
    MAPE_Lag.append(MAPE[0])


e = {'P-Value (Lag)':[1,5,10,15,25],'RMSE Errors':RMSE_Lag,'MAPE Errors':MAPE_Lag}
Errors = pd.DataFrame(e)
print(Errors)

print()
print()


plt.xticks([1,2,3,4,5],P)
plt.bar([1,2,3,4,5],RMSE_Lag)
plt.xlabel("Lagged Values")
plt.ylabel("RMSE Values")
plt.title("RMSE Values vs Lagged Values (p)")
plt.show()

plt.xticks([1,2,3,4,5],P)
plt.bar([1,2,3,4,5],MAPE_Lag)
plt.xlabel("Lagged Values")
plt.ylabel("MAPE Values")
plt.title("MAPE Values vs Lagged Values (p)")
plt.show()
    
print()
print()


#-------------------------------------------------------Question - 4-------------------------------------------------------#
print("#-------------------------------------------------------Question - 4-------------------------------------------------------#")


p = 1

while p < len(data):
    if(abs(np.corrcoef(train[p:].ravel(), train[:len(train)-p].ravel()))[0][1]<=2/math.sqrt(len(train[p:]))):
        print("The heuristic value for the optimal number of lags is:",p-1)
        break
    p += 1


print()
print()
    
p = p - 1

model = AR(train, lags=p, old_names=True)

model_fit = model.fit()

coef = model_fit.params

history = train[len(train)-p:]
history = [history[i] for i in range(len(history))]
predictions = list()

for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-p,length)]
    yhat = coef[0]
    for d in range(p):
        yhat += coef[d+1] * lag[p-d-1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)

T = len(test)
s1 = 0
s2 = 0

for t in range(T):
    diff1 = (predictions[t]-test[t])**2
    diff2 = (abs(test[t]-predictions[t]))/test[t]
    s1 += diff1
    s2 += diff2

num = np.sqrt(s1/T)
den = np.sum(test)/T

RMSE = (num/den)*100
MAPE = (s2/T)*100

print("RMSE Error is:",round(RMSE[0],3))
print("MAPE Error is:",round(MAPE[0],3))

print()
print()


#-------------------------------------------------------Extra Work-------------------------------------------------------#
print("#-------------------------------------------------------Extra Work-------------------------------------------------------#")


series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
X = series.values

p = 77

model = AR(X, lags=p, old_names=True)

model_fit = model.fit()

coef = model_fit.params

history = X[len(X)-p:]
history = [history[i] for i in range(len(history))]
predictions = list()

for i in range(117):
    lag = [history[i] for i in range(len(history)-p,len(history))]
    yhat = coef[0]
    
    for d in range(p):
        yhat += coef[d+1]*lag[p-d-1]
    predictions.append(yhat)
    history.append(yhat)

plt.figure(figsize=(20,10))
plt.xlabel("Days")
plt.ylabel("Predicted 3rd Peak")
plt.title("Line Plot for Predicted 3rd Wave")
plt.plot(predictions)
plt.show()