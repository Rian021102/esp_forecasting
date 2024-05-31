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
from keras.layers import Input
import tensorflow as tf
from keras import regularizers
from sklearn.decomposition import PCA


def load_data(path):
    df = pd.read_csv(path)
    print(df.info())
    #convert the date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    print(df.head())
    print(df.info())
    return df

def select_feat(df,feat_name):
    # Filter the DataFrame based on feature name
    df_feat = df[['Date', feat_name]]
    #sort the data based on date
    df_feat = df_feat.sort_values('Date')
    print(df_feat.head())
    #average the data based on date
    df_feat = df_feat.groupby('Date').mean().reset_index()
    #plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(df_feat['Date'], df_feat[feat_name])
    plt.title('Feature vs Time')
    plt.ylabel('Feature')
    plt.xlabel('Date')
    plt.show()
    return df_feat

def train_test (df_feat):
    #split based on length of the data with 80% training and 20% testing
    train_size = int(len(df_feat) * 0.65)
    test_size = len(df_feat) - train_size
    train, test = df_feat.iloc[0:train_size], df_feat.iloc[train_size:len(df_feat)]
    print(train.shape, test.shape)
    print(train.head())
    return train, test

def clean_train(train,feat_name):
    #drop all the missing values
    train = train.dropna()
    print(train.shape)
    print(train.head())
    #handling outliers using IQR
    #Q1 = train[feat_name].quantile(0.25)
    #Q3 = train[feat_name].quantile(0.75)
    #IQR = Q3 - Q1
    #lower_bound = Q1 - 1.5 * IQR
    #upper_bound = Q3 + 1.5 * IQR
    #train = train[(train[feat_name] > lower_bound) & (train[feat_name] < upper_bound)]
    # reduced the dimension of the data using PCA for the feature
    print(train.shape)
    return train

def clean_test(test):
    #drop all the missing values
    test = test.dropna()
    print(test.shape)
    print(test.head())
    return test

#plot the data as time series
def plot_data(train,feat_name):
    plt.figure(figsize=(12,6))
    plt.plot(train['Date'], train[feat_name])
    plt.title('Feature vs Time')
    plt.ylabel('Feature')
    plt.xlabel('Date')
    plt.show()

#preprocess train data to standardize the data
def preprocess_train(train, test, feat_name):
    # Create copies of the input DataFrames to avoid modifying original data
    scaled_train = train.copy()
    scaled_test = test.copy()

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit the scaler on the training data
    scaler = scaler.fit(train[[feat_name]])

    # Transform the 'Freq' column for both training and testing data
    scaled_train[feat_name] = scaler.transform(train[[feat_name]])
    scaled_test[feat_name] = scaler.transform(test[[feat_name]])

    print(scaled_train.head())
    return scaled_train, scaled_test, scaler


def create_sequences(data, time_steps):
    X = []
    for i in range(len(data) - time_steps):
        # Define the end of the sequence
        end_ix = i + time_steps
        # Gather input and output parts of the pattern
        seq_x = data[i:end_ix]
        X.append(seq_x)
    return np.array(X)

def create_data_test(scaled_train, scaled_test, time_steps, feat_name):
    # Convert DataFrame to numpy array for easier manipulation
    train_feat = scaled_train[feat_name].values
    test_feat = scaled_test[feat_name].values

    # Create sequences
    X_train = create_sequences(train_feat, time_steps)
    X_test = create_sequences(test_feat, time_steps)

    # Print the shapes to understand the dimensions
    print("Training data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)
    return X_train, X_test

def build_autoencoder(time_steps, n_features):
    with tf.device('/device:GPU:0'):

        # Encoder
        encoder = Sequential()
        encoder.add(LSTM(32, activation='relu', input_shape=(time_steps, n_features), return_sequences=False))
        encoder.add(RepeatVector(time_steps))  # This helps to repeat the context vector for the decoder

        # Decoder
        decoder = Sequential()
        decoder.add(LSTM(32, activation='relu', return_sequences=True))
        decoder.add(TimeDistributed(Dense(n_features)))  # Wraps Dense layer to output sequence

        # Autoencoder
        autoencoder = Sequential([encoder, decoder])
        autoencoder.compile(optimizer='adam', loss='mae')

        return autoencoder

def train_autoencoder(autoencoder, X_train, X_test,epochs=100,batch_size=32):
    # set early stopping criteria
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True
    )
    modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='/Users/rianrachmanto/miniforge3/project/esp_forecast_LSTM/model/autoencoder_ampere_large.h5',
        monitor='val_loss',
        save_best_only=True)
    
    history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, 
                              validation_data=(X_test, X_test), callbacks=[early_stopping, modelcheckpoint])
    return history

def plot_train_test_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

#inverse transform the reconstruction errors and the threshold
def inverse_transform(reconstruction_errors, threshold,predicted,scaler):
    # Inverse transform the reconstruction errors
    reconstruction_errors_inv = scaler.inverse_transform(reconstruction_errors.reshape(-1, 1)).flatten()
    # Inverse transform the threshold
    threshold_inv = scaler.inverse_transform(np.array([[threshold]]))
    # Inverse transform the predicted values
    predicted_inv = scaler.inverse_transform(predicted.reshape(-1, 1)).flatten()
    return reconstruction_errors_inv, threshold_inv, predicted_inv


def create_anomaly_df(test, reconstruction_errors_inv, threshold_inv, predicted, time_steps):
    # Adjusting the start index to skip the initial 'time_steps' entries
    test_score_df = test.iloc[time_steps:].copy()

    # Assigning loss, assuming reconstruction_errors_inv and test_score_df are now aligned
    test_score_df['loss'] = reconstruction_errors_inv

    # Handling threshold
    if threshold_inv.size == 1:
        test_score_df['threshold'] = np.full((len(test_score_df),), threshold_inv.flatten()[0])
    else:
        test_score_df['threshold'] = threshold_inv

    # Debug prints to check alignment
    print("Length of test_score_df:", len(test_score_df))
    print("Length of reconstruction_errors_inv:", len(reconstruction_errors_inv))
    print("Length of threshold_inv:", threshold_inv.size)
    print("Length of predicted:", predicted.shape[0])

    # Calculating anomaly flags
    test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']

    # Including raw predicted values ensuring they are aligned in length and indexing with the test_score_df
    if predicted.shape[0] == len(test_score_df):
        test_score_df['Predicted_Ampere'] = predicted[:, -1]  # assuming the last value in each prediction sequence
    else:
        # If there's a mismatch, log a warning and investigate the lengths
        print("Warning: Length of predicted values does not match the length of the DataFrame. Check alignment.")

    print(test_score_df.head())

    return test_score_df




# Adjust the function call appropriately or check the function call inputs as well.


   

def main():

    feat_name='Ampere'
    path='/Users/rianrachmanto/miniforge3/project/esp_new_02.csv'
    df=load_data(path)
    df_feat=select_feat(df,feat_name)
    train, test=train_test(df_feat)
    train=clean_train(train,feat_name)
    test=clean_test(test)
    plot_data(train,feat_name)
    scaled_train, scaled_test, scaler=preprocess_train(train, test, feat_name)
    time_steps=2
    n_features=1
    X_train, X_test=create_data_test(scaled_train, scaled_test, time_steps, feat_name)
    # Reshape the data for LSTM model
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print("Reshaped training data shape:", X_train.shape)
    print("Reshaped test data shape:", X_test.shape)

    autoencoder=build_autoencoder(time_steps,n_features)
    history=train_autoencoder(autoencoder, X_train, X_test,epochs=100,batch_size=32)
    plot_train_test_loss(history)

    predicted = autoencoder.predict(X_test)
    reconstruction_errors = np.mean(np.abs(predicted - X_test), axis=1)
    #estimate threshold from 95th percentile of reconstruction errors
    threshold = np.percentile(reconstruction_errors, 95)
    print("Threshold:", threshold)

    reconstruction_errors_inv, threshold_inv, predicted_inv = inverse_transform(reconstruction_errors, threshold, predicted, scaler)

    print(reconstruction_errors_inv)
    print(threshold_inv)
    print(predicted_inv)

    # Extract only the last prediction from each sequence if that's your model's structure
    predicted_flat = predicted.reshape(-1, predicted.shape[2])  # Assuming predicted is (num_samples, num_timesteps, num_features)
    predicted_last_step = predicted_flat[:, -1]
    predicted_inv = scaler.inverse_transform(predicted_last_step.reshape(-1, 1)).flatten()

    # Now call the function with properly aligned predicted_inv
    test_score_df = create_anomaly_df(test, reconstruction_errors_inv, threshold_inv, predicted, time_steps)

    corrected_predicted_values_inv = predicted_inv[-len(test_score_df):]

    # insert the corrected predicted values into the DataFrame
    test_score_df['Predicted_Volt'] = corrected_predicted_values_inv
    print(test_score_df.head())

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