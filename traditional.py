# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 04:15:27 2017

@author: Melwyn
"""

import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import pandas as pd

df=pd.read_csv('REL_qs.csv')
df = df[['Open',  'High',  'Low',  'Close']]

df.fillna(value=-99999, inplace=True)

print ('DF head')
print (df.head(10))
print ('DF tail')
print (df.tail(10))

forecast_out = 1

df['ForecastOpen'] = df['Open'].shift(-forecast_out)
df['ForecastClose'] = df['Close'].shift(-forecast_out)



X = np.array(df.drop(['ForecastOpen', 'ForecastClose'], 1))

#print(X[0], X.shape)
X = X[:-forecast_out]
#print(X[0], X.shape)
df.dropna(inplace=True)


y = np.array(df[['ForecastOpen']]) 
print (y[0])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


openModel = LinearRegression(n_jobs=-1)
openModel.fit(X_train, y_train)
confidence = openModel.score(X_test, y_test)
print("Accuracy Open TRADITIONAL: ", confidence * 100.0)


y = np.array(df[['ForecastClose']])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


closeModel = LinearRegression(n_jobs=-1)
closeModel.fit(X_train, y_train)
confidence = closeModel.score(X_test, y_test)
print ('Accuracy Close TRADITIONAL: ', confidence*100)

new_df = pd.DataFrame()
for i, row in df.iterrows():
    to_predict = [row.Open, row.High, row.Low, row.Close]
    temp_df = pd.DataFrame({'Open': [row.Open], 
                            'High': [row.High], 
                            'Low': [row.Low],
                            'Close': [row.Close],
                            'open_predicted': [openModel.predict(to_predict)[0][0]], 
                            'close_predicted': [closeModel.predict(to_predict)[0][0]]                                                
                            })
    new_df = pd.concat([new_df, temp_df])
    
new_df.to_csv('traditional.csv', sep=',', encoding='utf-8')