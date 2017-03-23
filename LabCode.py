# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 04:07:25 2017

@author: Melwyn
"""
import os
import math
#import quandl
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import csv

f = open('Report.txt', 'w')

def stock_price(company_name):
    
    df = pd.DataFrame.from_csv(company_name)
    df = df[['Open', 'High', 'Low', 'Close']]
    
    forecast_col = ['Open', 'High', 'Low', 'Close']

    df.fillna(value=-99999, inplace=True)
    
    days = 1
    df['ForecastOpen'] = df[forecast_col[0]].shift(-days)
    df['ForecastHigh'] = df[forecast_col[1]].shift(-days)
    df['ForecastLow'] = df[forecast_col[2]].shift(-days)
    df['ForecastClose'] = df[forecast_col[3]].shift(-days)
#    df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
#    df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    X = np.array(df.drop(['ForecastOpen', 'ForecastHigh', 'ForecastLow', 'ForecastClose'], 1))
    #X = preprocessing.scale(X)
    X = X[:-days]
    df.dropna(inplace=True)
    y = np.array(df[['ForecastOpen', 'ForecastHigh', 'ForecastLow', 'ForecastClose']])
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)

    print('Score: %f'%(confidence*100))
    f.write('Score: %f \n'%(confidence*100))
    last_row = df.head(1) 

    print(last_row)
    print ('Previous Day: ', last_row['Open'].values.item(0), last_row['High'].values.item(0), last_row['Low'].values.item(0), last_row['Close'].values.item(0))    
    s = ('Previous Day: '+  str(last_row['Open'].values.item(0)) + ' ' +
                            str(last_row['High'].values.item(0)) + ' ' + 
                            str(last_row['Low'].values.item(0)) + ' ' +
                            str(last_row['Close'].values.item(0)) + '\n')         
    f.write(s)
    
    Open = last_row['Open'].values.item(0)
    High = last_row['High'].values.item(0)
    Low = last_row['Low'].values.item(0)
    Close = last_row['Close'].values.item(0)
    #Hl_Pct = (High - Low) / Close * 100.0 
    #Pct_Change = (Close - Open) / Open * 100.0
    Predict = [[Open, High, Low, Close]]#, Hl_Pct, Pct_Change]
    
    return str(clf.predict(Predict))
   
def main():
    directory_path = os.path.dirname(__file__)+'\\Data'
    sectors = os.listdir(directory_path)
    for sector in sectors:
        company_list = os.listdir(directory_path+'\\'+sector)
        for company in company_list:
            print ('-----------------------------------------------------')
            print ('Company Name: '+company[4:len(company)-4])
            s = ('Company Name: '+company[4:len(company)-4]+'\n')            
            f.write (s)
            #print ('Current Day', stock_price(directory_path+'\\'+sector+'\\'+company))
            prediction = stock_price(directory_path+'\\'+sector+'\\'+company)
            s = ('Current Day: '+prediction+'\n')            
            f.write(s)            
            print ('-----------------------------------------------------')
#main()
            

pred = stock_price('C:\\Users\\Melwyn\\Desktop\\BEPROJECT\\Data\\Bank\\NSE-AXISBANK.csv')      
print (pred)            
            
f.close()

#    
#company_list = ['AXISBANK', 'BANKBARODA', 'HDFCBANK', 'INDUSINDBK']
#for i in range(4):
#    print(company_list[i])
#    print(stock_price('NSE-'+company_list[i]+'.csv'))
    
    
##csv_file = open('NSE-INFY.csv', 'r')
##csv_reader = csv.reader(csv_file)
##csv_list = list(csv_reader)
##
##length = len(csv_list)
##
##df = pd.DataFrame(columns=('Open', 'High', 'Low', 'Lasr', 'Close', 'Total Trade Quantity', 'Turnover (Lacs)'))
##
##for i in range(length-1):
##    #print(csv_list[i+1][1:])
##    df.loc[i+1] = list(map(float, csv_list[i+1][1:]))
#
#
#df = pd.DataFrame.from_csv('NSE-INFY.csv')
#
#
##df = quandl.get("NSE/INFY")
#df = df[['Open',  'High',  'Low',  'Close']]
#
#df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
#df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
#
#print(df.tail())
#
#
#forecast_col = ['Open',  'High',  'Low',  'Close']
#df.fillna(value=-99999, inplace=True)
#
## forecast_out basically the days ka gap u want to set
#forecast_out = 1
#
#df['ForecastOpen'] = df[forecast_col[0]].shift(-forecast_out)
#df['ForecastHigh'] = df[forecast_col[1]].shift(-forecast_out)
#df['ForecastLow'] = df[forecast_col[2]].shift(-forecast_out)
#df['ForecastClose'] = df[forecast_col[3]].shift(-forecast_out)
#
#
#
#X = np.array(df.drop(['ForecastOpen', 'ForecastHigh', 'ForecastLow', 'ForecastClose'], 1))
#print(X[0], X.shape)
#
#X = X[:-forecast_out]
#print(X[0], X.shape)
#df.dropna(inplace=True)
#
#
#y = np.array(df[['ForecastOpen', 'ForecastHigh', 'ForecastLow', 'ForecastClose']])
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
#
#print("y: ", y[len(y) - 1])
#print("X_test: ", X_test[len(X_test) - 1])
#
#clf = LinearRegression(n_jobs=-1)
#clf.fit(X_train, y_train)
#confidence = clf.score(X_test, y_test)
#print("Accuracy: ", confidence * 100.0)
#
#
#print(df.tail(1))
#
#
## Prediction
#last_row = df.head(1) 
#
#Open = last_row['Open'].values.item(0)#940.0
#High = last_row['High'].values.item(0)#945.0
#Low = last_row['Low'].values.item(0)#918.7
#Close = last_row['Close'].values.item(0)#921.55
#Hl_Pct = (High - Low) / Close * 100.0 
#Pct_Change = (Close - Open) / Open * 100.0
##Precict = preprocessing.scale([[Open, High, Low, Close, Hl_Pct, Pct_Change]])
#Precict = [[Open, High, Low, Close, Hl_Pct, Pct_Change]]
##print("Predict: ", Precict)
#
#
#print('Previous Open', Open, 'Previous Close', Close)
#
#
#
#print("Open", "High", "Low", "Close")
#print(clf.predict(Precict))