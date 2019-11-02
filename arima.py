# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:09:47 2019

@author: Asus
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
BU=pd.read_csv('USD_BYN.csv', index_col='Date',parse_dates=True)
fig, ax = plt.subplots()
BU.plot(ax=ax)
plt.show()
#There is a trend on the plot
# Splitting the data into a train and test set
BU_train = BU.loc[:'2017']
BU_test = BU.loc['2018':]
fig, ax = plt.subplots()
BU_train.plot(ax=ax)
BU_test.plot(ax=ax)
plt.show()
from statsmodels.tsa.stattools import adfuller
result = adfuller(BU['Close'])
#test statistic
print(result[0])
#T>0
# p-value 
print(result[1])
#p>0.05 we can't reject null hypothesis (non-stationary)
#critical values
print(result[4]) 
#Taking the first differece to make the model stationry
BU_1 = BU[['Close']]
print(BU_1.head())
BU_diff=BU_1.diff().dropna()
result_diff = adfuller(BU_diff['Close'])
#T-stat. <-2 and p-value is close to zero - we reject null hypothesis
print(result_diff)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
plot_acf(BU_1, lags=10, zero=False, ax=ax1)
plot_pacf(BU_1, lags=10, zero=False, ax=ax2)
plt.show()
#AR
from statsmodels.tsa.arima_model import ARMA
model = ARMA(BU['Close'], order=(1,0))
results = model.fit()
print(results.summary())
#ar.L1.Close has p-value close to 0 and is statistically sagnificant
from statsmodels.tsa.statespace.sarimax import SARIMAX
model=SARIMAX(BU_1, order=(1,0,0), trend='c')
results = model.fit()
print(results.summary())
results.plot_diagnostics()
"""JB p-value shows that the residuals of the model are not normally distributed
 and they are not correlated according to Prob(Q) """ 
plt.show()
plt.close()
mae = np.mean(np.abs(results.resid))
print(mae)
one_step_forecast = results.get_prediction(start=-12)
mean_forecast = one_step_forecast.predicted_mean
confidence_intervals = one_step_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower Close']
upper_limits = confidence_intervals.loc[:,'upper Close']
print(mean_forecast)
plt.plot(BU_1.index, BU_1, label='observed')
plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')
# shading the area between confidence limits
plt.fill_between(lower_limits.index, lower_limits, 
               upper_limits, color='pink')
plt.xlabel('Date')
plt.ylabel('USD-BYN')
plt.legend()
plt.show()
plt.close()
#Dynamic forecast
dynamic_forecast = results.get_prediction(start=-12, dynamic=True)
mean_forecastd = dynamic_forecast.predicted_mean
confidence_intervalsd = dynamic_forecast.conf_int()
lower_limitsd = confidence_intervalsd.loc[:,'lower Close']
upper_limitsd = confidence_intervalsd.loc[:,'upper Close']
print(mean_forecastd)
plt.plot(BU_1.index, BU_1, label='observed')
plt.plot(mean_forecastd.index, mean_forecastd, color='r', label='forecast')
plt.fill_between(lower_limitsd.index, lower_limitsd, 
               upper_limitsd, color='pink')
plt.xlabel('Date')
plt.ylabel('USD-BYN')
plt.legend()
plt.show()
plt.close()
