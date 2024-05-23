from keras.layers import LSTM
from keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#standardize the data
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

'''
This is for performing anomaly detection on the voltage data, using trained model
'''
def data_pipeline(path):
    # load the data
    df=pd.read_csv(path)
    # convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def filter_data(df, column_name, well_name):
    # filter the data based on the well name and column name. No need to flter the Date
    # Filter the DataFrame for the given well name
    df_filtered = df[df['Well'] == well_name][['Well', 'Date', column_name]]
    print(df_filtered.head())
    return df_filtered

def preprocess_data(df_filtered,column_name):
    # standardize the data
    scaler = StandardScaler()
    # dropna
    df_filtered = df_filtered.dropna()
    # remove Well column
    print(df_filtered.head())
    df_filtered = df_filtered.drop(columns=['Well'])
    # standardize the data
    df_filtered[column_name] = scaler.fit_transform(df_filtered[column_name].values.reshape(-1,1))
    return df_filtered

def create_sequences(data, time_steps):
    # create sequences
    X = []
    for i in range(len(data) - time_steps):
        # Define the end of the sequence
        end_ix = i + time_steps
        # Gather input and output parts of the pattern
        seq_x = data[i:end_ix]
        X.append(seq_x)
    return np.array(X)

def create_dataset(df_filtered, time_steps,column_name):
    # create sequences
    data = df_filtered[column_name].values
    X = create_sequences(data, time_steps)
    return X
def main():
    path='/Users/rianrachmanto/miniforge3/project/esp_new.csv'
    model= tf.keras.models.load_model('/Users/rianrachmanto/miniforge3/project/esp_forecast_LSTM/model/autoencoder_voltage.h5')
    df=data_pipeline(path)
    column_name='Volt'
    well_name='BS3'
    time_steps=3
    df_filtered=filter_data(df, column_name, well_name)
    df_filtered=preprocess_data(df_filtered,column_name)
    X=create_dataset(df_filtered, time_steps,column_name)
    # make a prediction
    yhat = model.predict(X)
    # reshape the data
    X = X.reshape((X.shape[0], X.shape[1]))

if __name__ == '__main__':
    main()


    