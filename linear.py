# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 05:14:49 2017

@author: Melwyn
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn import preprocessing, cross_validation

def stock_predict(company_name):
    df = pd.DataFrame.from_csv(company_name)
    df = df[['Open', 'High', 'Low', 'Close']]
    
    df['ForecastOpen'] = df['Open'].shift(-1)
    df['ForecastHigh'] = df['High'].shift(-1)
    df['ForecastLow'] = df['Low'].shift(-1)
    df['ForecastClose'] = df['Close'].shift(-1)
    
    df['HL_PCT'] = ((df['High'] - df['Low']) / df['Close']) * 100.0
    df['PCT_change'] = ((df['Close'] - df['Open']) / df['Open']) * 100.0
    
    X = np.array(df.drop(['ForecastOpen', 'ForecastHigh', 'ForecastLow', 'ForecastClose'], 1))
    X = preprocessing.scale(X)
    X = X[:-1]
    df.dropna(inplace=True)
    
    y = np.array(df[['ForecastOpen', 'ForecastHigh', 'ForecastLow', 'ForecastClose']])    
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    
    
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    
    print (accuracy)
    
stock_predict('C:\\Users\\Melwyn\\Desktop\\BEPROJECT\\Data\\Bank\\NSE-AXISBANK.csv')