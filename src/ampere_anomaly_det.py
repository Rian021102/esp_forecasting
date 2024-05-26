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
    return df_filtered,scaler

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
    path = '/Users/rianrachmanto/miniforge3/project/esp_new.csv'
    model = tf.keras.models.load_model('/Users/rianrachmanto/miniforge3/project/esp_forecast_LSTM/model/autoencoder_ampere.h5')
    df = data_pipeline(path)
    column_name = 'Ampere'
    well_name = 'MHN-6'
    df_filtered = filter_data(df, column_name, well_name)  # Only gets DataFrame
    df_filtered, scaler = preprocess_data(df_filtered, column_name)  # Gets DataFrame and scaler
    time_steps = 2  # Adjusted to match the model's expected input shape
    X = create_dataset(df_filtered, time_steps, column_name)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape X to include the feature dimension
    yhat = model.predict(X)
    reconstruction_error = np.mean(np.abs(yhat - X), axis=1)
    threshold = np.percentile(reconstruction_error, 95)
    reconstruction_errors_inv = scaler.inverse_transform(reconstruction_error.reshape(-1, 1)).flatten()
    threshold_inv = scaler.inverse_transform(np.array([[threshold]]))
    predicted_inv = scaler.inverse_transform(yhat.reshape(-1, 1)).flatten()
    test_score_df = df_filtered.iloc[time_steps:].copy()

    # Invert the scaled 'Volt' values back to original scale
    test_score_df['Ampere'] = scaler.inverse_transform(test_score_df[column_name].values.reshape(-1, 1))

    test_score_df['loss'] = reconstruction_errors_inv
    test_score_df['threshold'] = np.full(len(test_score_df), threshold_inv[0])
    test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
    if yhat.shape[0] == len(test_score_df):
        test_score_df['Predicted_Ampere'] = predicted_inv[-len(test_score_df):]
    else:
        print("Warning: Length of predicted values does not match the length of the DataFrame. Check alignment.")
    print(test_score_df.head())

    #print number of anomalies
    print(test_score_df['anomaly'].value_counts())

    # Plot actual, predicted, threshold and anomaly
    plt.figure(figsize=(12, 6))
    plt.plot(test_score_df['Date'], test_score_df['Ampere'], color='blue', label='Actual')
    plt.plot(test_score_df['Date'], test_score_df['Predicted_Ampere'], color='red', label='Predicted')
    plt.scatter(test_score_df.loc[test_score_df['anomaly'], 'Date'], test_score_df.loc[test_score_df['anomaly'], 'Ampere'], color='red', label='Anomaly')
    plt.title('Anomaly Detection')
    plt.ylabel('Ampere')
    plt.xlabel('Date')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()








    