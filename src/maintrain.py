import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def datapipe(path):

    df = pd.read_csv(path)
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.WELL_ID.unique())
    print(df.shape)
    #create BLPD from BOPD plus BWPD
    df['BLPD'] = df['BOPD'] + df['BWPD']
    return df

def edit_data(df):
    #sort data by date
    dfed = df.sort_values(by='DATE', ascending=True)
   #lineplot as timeseries with BLPD and current with BLPD as secondary y axis
    plt.figure(figsize=(20,10))
    sns.lineplot(x='DATE', y='BLPD', data=dfed)
    plt.ylabel('BLPD')
    plt.twinx()
    sns.lineplot(x='DATE', y='CURRENT', data=dfed, color='red')
    plt.ylabel('CURRENT')
    #show legend
    plt.legend(['BLPD', 'CURRENT'])
    plt.show()
    
    return dfed

def splitdata(dfed):
    #split data using length of data where train data should be 70% of the data
    train_size = int(len(dfed)*0.7)
    train = dfed[:train_size]
    test = dfed[train_size:]
    print(train.shape, test.shape)
    return train, test

def feature_target(train, test):
    #only pick DATE, CURRENT and BLPD
    train_feat= train[['DATE', 'CURRENT', 'BLPD']]
    test_feat = test[['DATE', 'CURRENT', 'BLPD']]
    return train_feat, test_feat

def eda(train_feat):
    #check for missing values
    print(train_feat.isnull().sum())
    #check for correlation between features
    print(train_feat.corr())
    #plot correlation heatmap
    plt.figure(figsize=(10,10))
    sns.heatmap(train_feat.corr(), annot=True)
    plt.show()



def main():
    path = '/Users/rianrachmanto/miniforge3/project/esp_forecast_LSTM/data/wells_data_final.csv'
    df = datapipe(path)
    dfed = edit_data(df)
    train, test = splitdata(dfed)
    train_feat, test_feat = feature_target(train, test)
    eda(train_feat)

if __name__ == '__main__':
    main()

