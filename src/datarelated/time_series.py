import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def read_data(path):
    df = pd.read_csv(path)
    df.replace('No Data', np.nan, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.dropna(inplace=True)
    numeric_columns = ['Frequency', 'Voltage', 'Ampere', 'Pressure_Discharge', 'Pressure_Intake', 'Temp_Intake', 'Temp_Motor', 'Vibration_X', 'Vibration_Y']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    print(df.shape)
    print(df.isna().sum())
    print(df.Well_ID.unique())
    #sort by well id and date
    df.sort_values(by=['Well_ID', 'Date'], inplace=True)
    return df

def volatage_analysis(df):
    df_bs3 = df[df['Well_ID'] == 'YWA20']
    #plot time series of voltage
    plt.plot(df_bs3['Date'], df_bs3['Voltage'])
    plt.show()

def main():
    path = '/Users/rianrachmanto/pypro/data/data_esp_edit02.csv'
    df = read_data(path)
    volatage_analysis(df)

if __name__ == '__main__':
    main()