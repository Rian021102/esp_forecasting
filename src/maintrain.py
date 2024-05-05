import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.metrics import MeanAbsoluteError
import tensorflow as tf

def datapipe(path):
    df = pd.read_csv(path)
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.WELL_ID.unique())
    print(df.shape)
    df['BLPD'] = df['BOPD'] + df['BWPD']  # Assume BOPD and BWPD are in your CSV
    return df

def edit_data(df):
    dfed = df.sort_values(by='DATE', ascending=True)
    plt.figure(figsize=(20,10))
    sns.lineplot(x='DATE', y='BLPD', data=dfed)
    plt.ylabel('BLPD')
    ax2 = plt.twinx()
    sns.lineplot(x='DATE', y='CURRENT', data=dfed, color='red')
    ax2.set_ylabel('CURRENT')
    plt.legend(['BLPD', 'CURRENT'])
    plt.show()
    return dfed

def splitdata(dfed):
    train_size = int(len(dfed)*0.7)
    train = dfed[:train_size]
    test = dfed[train_size:]
    print(train.shape, test.shape)
    return train, test

def feature_target(train, test):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[['CURRENT']])
    test_scaled = scaler.transform(test[['CURRENT']])
    return train_scaled, test_scaled, scaler

def create_dataset(X, time_steps=1):
    Xs = []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps), 0]
        Xs.append(v)
    return np.array(Xs)

def lstm_model(input_shape):
    with tf.device('/CPU:0'):
            model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(input_shape, 1)),
            LSTM(50),
            Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanAbsoluteError()])
            return model

def plot_history(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    path = '/Users/rianrachmanto/miniforge3/project/esp_forecast_LSTM/data/wells_data_final.csv'
    df = datapipe(path)
    dfed = edit_data(df)
    dfed['DATE'] = pd.to_datetime(dfed['DATE'])
    dfed.set_index('DATE', inplace=True)
    train, test = splitdata(dfed)
    train_feat, test_feat, scaler = feature_target(train, test)
    time_steps = 10
    X_train = create_dataset(train_feat, time_steps)
    X_test = create_dataset(test_feat, time_steps)
    model = lstm_model(time_steps)
    history = model.fit(X_train, X_train, epochs=30, batch_size=32, validation_split=0.1, verbose=1)
    plot_history(history)
    model.save('lstm_model.h5')  # Save the model
    y_pred = model.predict(X_test)
    
    # Using Seaborn to plot the predictions
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=np.arange(len(X_test)), y=X_test[:, 0], label='True')
    sns.lineplot(x=np.arange(len(y_pred)), y=y_pred[:, 0], label='Predicted')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
