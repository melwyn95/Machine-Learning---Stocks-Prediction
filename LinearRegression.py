# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:42:03 2016

@author: Melwyn
"""
import quandl
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression


df = quandl.get("NSE/INFY")
df = df[['Open',  'High',  'Low',  'Close']]

df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

print(df.tail())


forecast_col = ['Open',  'High',  'Low',  'Close']
df.fillna(value=-99999, inplace=True)

# forecast_out basically the days ka gap u want to set
forecast_out = 1

df['ForecastOpen'] = df[forecast_col[0]].shift(-forecast_out)
df['ForecastHigh'] = df[forecast_col[1]].shift(-forecast_out)
df['ForecastLow'] = df[forecast_col[2]].shift(-forecast_out)
df['ForecastClose'] = df[forecast_col[3]].shift(-forecast_out)



X = np.array(df.drop(['ForecastOpen', 'ForecastHigh', 'ForecastLow', 'ForecastClose'], 1))
print(X[0], X.shape)

X = X[:-forecast_out]
print(X[0], X.shape)
df.dropna(inplace=True)


y = np.array(df[['ForecastOpen', 'ForecastHigh', 'ForecastLow', 'ForecastClose']])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

print("y: ", y[len(y) - 1])
print("X_test: ", X_test[len(X_test) - 1])

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print("Accuracy: ", confidence * 100.0)


# Prediction 
Open = 940.0
High = 945.0
Low = 918.7
Close = 921.55
Hl_Pct = (High - Low) / Close * 100.0 
Pct_Change = (Close - Open) / Open * 100.0
#Precict = preprocessing.scale([[Open, High, Low, Close, Hl_Pct, Pct_Change]])
Precict = [[Open, High, Low, Close, Hl_Pct, Pct_Change]]
#print("Predict: ", Precict)
print("Open", "High", "Low", "Close")
print(clf.predict(Precict))
