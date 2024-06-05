from keras.layers import LSTM
from keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import streamlit as st

st.title('ESP Anomaly Detection')
st.header('Leveraging LSTM Autoencoder for Anomaly Detection')

def preprocess_data(df, column_name):
    scaler = StandardScaler()
    df = df.dropna()
    df = df.drop(columns=['Well'])  # Ensure this column exists
    df[column_name] = scaler.fit_transform(df[column_name].values.reshape(-1,1))
    return df, scaler

def create_sequences(data, time_steps):
    X = []
    for i in range(len(data) - time_steps):
        end_ix = i + time_steps
        seq_x = data[i:end_ix]
        X.append(seq_x)
    return np.array(X)

def create_dataset(df, time_steps, column_name):
    data = df[column_name].values
    X = create_sequences(data, time_steps)
    return X

# Paths to the models
model_amp = '/Users/rianrachmanto/miniforge3/project/esp_forecast_LSTM/model/autoencoder_ampere_no_iqr_knn_imputer.h5'
model_vol = '/Users/rianrachmanto/miniforge3/project/esp_forecast_LSTM/model/autoencoder_voltage.h5'
model_temp = '/Users/rianrachmanto/miniforge3/project/esp_forecast_LSTM/model/autoencoder_temp.h5'

st.subheader('Load Data')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    df['Date'] = pd.to_datetime(df['Date'])
    df.drop(columns=['Hours_Online', 'Gross_Rate'], inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].astype(float)
    well_name = st.sidebar.selectbox('Select Well Name', df['Well'].unique())
    features = ['Ampere', 'Volt', 'TM']
    time_steps = 2

    for col in features:
        df_feat = df[df['Well'] == well_name][['Well', 'Date', col]]
        df_feat, scaler = preprocess_data(df_feat, col)
        X = create_dataset(df_feat, time_steps, col)

        if col == 'Ampere':
            model = tf.keras.models.load_model(model_amp)
        elif col == 'Volt':
            model = tf.keras.models.load_model(model_vol)
        else:
            model = tf.keras.models.load_model(model_temp)

        yhat = model.predict(X)
        yhat = yhat.squeeze(-1)  # Adjust shape to match X
        reconstruction_error = np.mean(np.abs(yhat - X), axis=1)
        threshold = np.percentile(reconstruction_error, 95)
        reconstruction_errors_inv = scaler.inverse_transform(reconstruction_error.reshape(-1, 1)).flatten()
        threshold = np.percentile(reconstruction_error, 95)
        reconstruction_errors_inv = scaler.inverse_transform(reconstruction_error.reshape(-1, 1)).flatten()
        threshold_inv = scaler.inverse_transform(np.array([[threshold]]))

        # Since threshold_inv is 2D array, use .item() to extract the value as scalar
        test_score_df = df_feat.iloc[time_steps:].copy()
        test_score_df[col] = scaler.inverse_transform(test_score_df[col].values.reshape(-1, 1))
        test_score_df['Loss'] = reconstruction_errors_inv
        test_score_df['Threshold'] = threshold_inv.item() * np.ones(len(test_score_df))  # Set threshold for all rows
        test_score_df['Anomaly'] = test_score_df['Loss'] > test_score_df['Threshold']

        st.write(test_score_df)

        plt.figure(figsize=(12, 6))
        plt.plot(test_score_df['Date'], test_score_df[col], label='Data', zorder=1)
        plt.scatter(test_score_df['Date'][test_score_df['Anomaly']], test_score_df[col][test_score_df['Anomaly']], color='red', label='Outliers', zorder=2)
        plt.title(f'{well_name} - {col} with Outliers')
        plt.legend()
        st.pyplot(plt)
