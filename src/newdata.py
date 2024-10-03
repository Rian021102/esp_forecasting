import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
    df=pd.read_csv(path)
    return df

def main():
    path='/Users/rianrachmanto/pypro/project/esp_forecasting/data/wells_data_final.csv'
    df=load_data(path)
    print(df.head())
    print(df.columns)
    print(len(df))
    #check for missing values
    print(df.isnull().sum().sum())
if __name__ == '__main__':
    main()