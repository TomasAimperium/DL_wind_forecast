import pandas as pd
from scipy.signal import savgol_filter
import numpy as np


def savgol(X):
    savgol_data = savgol_filter(X, 21, 1)
    return savgol_data




def filter_agg(X,station):


    data = X.reset_index(drop = True)
    
    data['Meteo Station 04 - Wind Speed(m/s)'] = data['Meteo Station 04 - Wind Speed(m/s)'].apply(lambda x : 0 if x<-1000 else x)
    data['Meteo Station 04 - Wind Direction(º)'] = data['Meteo Station 04 - Wind Direction(º)'].apply(lambda x : 0 if x<-1000 else x)
    data['Meteo Station 04 - Wind Direction Rad(rad)'] = data['Meteo Station 04 - Wind Direction Rad(rad)'].apply(lambda x : 0 if x<-10 else x)
    data['Meteo Station 04 - Atmospheric Pressure(mB)'] = data['Meteo Station 04 - Atmospheric Pressure(mB)'].apply(lambda x : 887.82 if x<500 else x)
    data['Meteo Station 04 - External Ambient Temperature(ºC)'] = data['Meteo Station 04 - External Ambient Temperature(ºC)'].apply(lambda x : 0 if x<-1000 else x)
    data['Meteo Station 04 - Humidity(%)'] = data['Meteo Station 04 - Humidity(%)'].apply(lambda x : 0 if x<-1000 else x)
    data['Meteo Station 10 - Wind Direction(º)'] = data['Meteo Station 10 - Wind Direction(º)'].apply(lambda x : 0 if x<-1000 else x)
    data['Meteo Station 10 - Wind Speed(m/s)'] = data['Meteo Station 10 - Wind Speed(m/s)'].apply(lambda x : 0 if x<-1000 else x)
    data['Meteo Station 10 - Wind Direction Rad(rad)'] = data['Meteo Station 10 - Wind Direction Rad(rad)'].apply(lambda x : 0 if x<-10 else x)
    data['Datetime'] =  pd.to_datetime(data['Datetime'], format='%Y-%m-%d %H:%M:%S')
    data_agg = data.resample('5Min', on='Datetime').mean()
    data_noNa = data_agg.dropna()

    target = 'Meteo Station '+ station +' - Wind Speed(m/s)'
    y = data_noNa.loc[:,[target]].values
     
    
    return pd.Series(y.reshape(len(y)))



def agg(X):
    data = X
    data['Datetime'] =  pd.to_datetime(data['Datetime'], format='%Y-%m-%d %H:%M:%S')
    data_agg = data.resample('5Min', on='Datetime').mean()
    y = data_agg.dropna()
    return  y




def prepro(X):
    Y = agg(X)
    savgol_data = pd.DataFrame()

    
    for col in Y.columns:
        savgol_data[col] = savgol_filter(Y[col], 21, 1)


    return savgol_data     




