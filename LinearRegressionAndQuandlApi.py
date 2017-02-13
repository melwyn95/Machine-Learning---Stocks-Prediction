# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:42:03 2016

@author: Melwyn
"""
import quandl
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression

df = quandl.get("NSE/INFY", authtoken="-zLmnBx6NmesMSEA_2MU")#Quandl.get("WIKI/GOOGL")
#df = df[['Open',  'High',  'Low',  'Close', 'Total Trade Quantity']]
df = df[['Open',  'High',  'Low',  'Close', 'Last']]
df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

print("Before----")
print(df.tail())

#df = df[['Close', 'HL_PCT', 'PCT_change', 'Total Trade Quantity']]
#forecast_col = 'Adj. Close'
#forecast_col = ['Open',  'High',  'Low',  'Close', 'Total Trade Quantity']
forecast_col = ['Open',  'High',  'Low',  'Close', 'Last']
df.fillna(value=-99999, inplace=True)
#forecast_out = int(math.ceil(0.01 * len(df)))
forecast_out = 1
df['ForecastOpen'] = df[forecast_col[0]].shift(-forecast_out)
df['ForecastHigh'] = df[forecast_col[1]].shift(-forecast_out)
df['ForecastLow'] = df[forecast_col[2]].shift(-forecast_out)
df['ForecastClose'] = df[forecast_col[3]].shift(-forecast_out)
df['ForecastLast'] = df[forecast_col[4]].shift(-forecast_out)
#df['ForecastTTQ'] = df[forecast_col[4]].shift(-forecast_out)

print("After----")
print(df.tail())

#X = np.array(df.drop(['ForecastOpen', 'ForecastHigh', 'ForecastLow', 'ForecastClose', 'ForecastTTQ'], 1))
X = np.array(df.drop(['ForecastOpen', 'ForecastHigh', 'ForecastLow', 'ForecastClose', 'ForecastLast'], 1))
print(X[0], X.shape)
#X = preprocessing.scale(X)
X = X[:-forecast_out]
print(X[0], X.shape)
df.dropna(inplace=True)

#print(df['ForecastOpen', 'ForecastHigh', 'ForecastLow', 'ForecastClose', 'ForecastTTQ'].head())


#print(df[['ForecastOpen', 'ForecastHigh', 'ForecastLow', 'ForecastClose', 'ForecastTTQ']].head())

#y = np.array(df[['ForecastOpen', 'ForecastHigh', 'ForecastLow', 'ForecastClose', 'ForecastTTQ']])
y = np.array(df[['ForecastOpen', 'ForecastHigh', 'ForecastLow', 'ForecastClose', 'ForecastLast']])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

print("y: ", y[len(y) - 1])
print("X_test: ", X_test[len(X_test) - 1])

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print("Accuracy: ", confidence * 100.0)


# Prediction 
Open = 995.05
High = 1004.20
Low = 992.25
Close = 997.75
Last = 992.45
Hl_Pct = (High - Low) / Close * 100.0 
Pct_Change = (Close - Open) / Open * 100.0
#Precict = preprocessing.scale([[Open, High, Low, Close, Hl_Pct, Pct_Change]])
Precict = [[Open, High, Low, Close, Last, Hl_Pct, Pct_Change]]
#print("Predict: ", Precict)
print("Open", "High", "Low", "Close", "Last")
print(clf.predict(Precict))
