# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 05:47:32 2017

@author: Melwyn
"""

import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import pandas as pd

df=pd.read_csv('REL_qs.csv')
df = df[['Open',  'High',  'Low',  'Close', 'open_score', 'close_score']]
forecast_col = ['Open',  'High',  'Low',  'Close']
df.fillna(value=-99999, inplace=True)

print ('DF head')
print (df.head(10))
print ('DF tail')
print (df.tail(10))

forecast_out = 1

df['ForecastOpen'] = df[forecast_col[0]].shift(-forecast_out)
df['ForecastClose'] = df[forecast_col[3]].shift(-forecast_out)




X = np.array(df.drop(['ForecastOpen', 'ForecastClose', 'close_score'], 1))
X = X[:-forecast_out]
df.dropna(inplace=True)


y = np.array(df[['ForecastOpen']]) 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

openModel = LinearRegression(n_jobs=-1)
openModel.fit(X_train, y_train)
confidence = openModel.score(X_test, y_test)
print("Accuracy Open: ", confidence * 100.0)

X = np.array(df.drop(['ForecastOpen', 'ForecastClose', 'open_score'], 1))
y = np.array(df[['ForecastClose']])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

closeModel = LinearRegression(n_jobs=-1)
closeModel.fit(X_train, y_train)
confidence = closeModel.score(X_test, y_test)
print ('Accuracy Close: ', confidence*100)


new_df = pd.DataFrame()
for i, row in df.iterrows():
    to_predict_open = [row.Open, row.High, row.Low, row.Close, row.open_score]
    to_predict_close = [row.Open, row.High, row.Low, row.Close, row.close_score]
    temp_df = pd.DataFrame({'Open': [row.Open], 
                            'High': [row.High], 
                            'Low': [row.Low],
                            'Close': [row.Close],
                            'open_score': [row.open_score],
                            'close_score': [row.close_score],
                            'open_predicted': [openModel.predict(to_predict_open)[0][0]], 
                            'close_predicted': [closeModel.predict(to_predict_close)[0][0]]                                                
                            })
    new_df = pd.concat([new_df, temp_df])
    
new_df.to_csv('hybrid.csv', sep=',', encoding='utf-8')