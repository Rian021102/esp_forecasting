import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
import tensorflow as tf

def load_data(path, features):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df[['Date', 'Well'] + features]

def select_well(df, well):
    return df[df['Well'] == well]

def drop_na_values(df):
    return df.dropna()

def train_test_split(df, split_ratio=0.7):
    train_size = int(len(df) * split_ratio)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    return train, test

def scale_data(train, test, features):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[features])
    test_scaled = scaler.transform(test[features])
    train[features] = train_scaled
    test[features] = test_scaled
    return train, test, scaler

def create_sequences(data, time_steps, feature_columns):
    X = []
    for i in range(len(data) - time_steps):
        seq_x = data[feature_columns].iloc[i:i + time_steps].values
        X.append(seq_x)
    return np.array(X)

def build_autoencoder(time_steps, n_features):
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(time_steps, n_features), return_sequences=False))
    model.add(RepeatVector(time_steps))
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(optimizer='adam', loss='mae')
    return model

def train_autoencoder(autoencoder, X_train, X_test, epochs=100, batch_size=32):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True
    )
    modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='/Users/rianrachmanto/miniforge3/project/esp_forecast_LSTM/model/multi_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
    history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, 
                              validation_data=(X_test, X_test), callbacks=[early_stopping, modelcheckpoint])
    return history

def plot_train_test_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

def predict_and_evaluate(autoencoder, X_test):
    predicted = autoencoder.predict(X_test)
    reconstruction_errors = np.mean(np.abs(predicted - X_test), axis=1)
    threshold = np.percentile(reconstruction_errors, 95)
    return predicted, reconstruction_errors, threshold

def inverse_transform(scaler, test, features, time_steps, reconstruction_errors, threshold, predicted):
    predicted_inv = scaler.inverse_transform(predicted.reshape(-1, len(features)))
    reconstruction_errors_inv = scaler.inverse_transform(reconstruction_errors.reshape(-1, 1)).flatten()
    threshold_inv = scaler.inverse_transform(np.array([[threshold]]))[0][0]

    test_adjusted = test.iloc[time_steps:]
    test_adjusted['Predicted'] = np.mean(predicted_inv, axis=1)
    test_adjusted['Reconstruction Error'] = reconstruction_errors_inv
    test_adjusted['Threshold'] = threshold_inv
    test_adjusted['Anomaly'] = test_adjusted['Reconstruction Error'] > threshold_inv
    return test_adjusted

def plot_results(test_adjusted):
    plt.figure(figsize=(14, 6))
    plt.plot(test_adjusted['Date'], test_adjusted.iloc[:, 2:6].mean(axis=1), label='Actual Values')
    plt.plot(test_adjusted['Date'], test_adjusted['Predicted'], 'r', label='Predicted Values')
    anomalies = test_adjusted[test_adjusted['Anomaly'] == True]
    plt.scatter(anomalies['Date'], anomalies.iloc[:, 2:6].mean(axis=1), color='k', label='Anomaly')
    plt.legend()
    plt.show()

# Example usage
path = '/Users/rianrachmanto/miniforge3/project/esp_new_02.csv'
features = ['Ampere', 'Volt', 'Vibration', 'TM']
well = 'YWB-15'
df = load_data(path, features)
df = select_well(df, well)
df = drop_na_values(df)
train, test = train_test_split(df)
train, test, scaler = scale_data(train, test, features)

time_steps = 24
X_train = create_sequences(train, time_steps, features)
X_test = create_sequences(test, time_steps, features)

X_train = X_train.reshape((X_train.shape[0], time_steps, len(features)))
X_test = X_test.reshape((X_test.shape[0], time_steps, len(features)))

autoencoder = build_autoencoder(time_steps, len(features))
history = train_autoencoder(autoencoder, X_train, X_test)

plot_train_test_loss(history)

predicted, reconstruction_errors, threshold = predict_and_evaluate(autoencoder, X_test)
test_adjusted = inverse_transform(scaler, test, features, time_steps, reconstruction_errors, threshold, predicted)
plot_results(test_adjusted)
