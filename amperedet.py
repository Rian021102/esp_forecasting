import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import requests
import json

FASTAPI_URL = 'http://127.0.0.1:8000/predict/'

def preprocess_data(df, column_name):
    scaler = StandardScaler()
    # Ensure data is a float type for standardization
    df[column_name] = df[column_name].astype(float)
    df[column_name] = scaler.fit_transform(df[[column_name]])
    return df, scaler

def create_sequences(data, time_steps):
    X = []
    for i in range(len(data) - time_steps + 1):
        seq_x = data[i:i + time_steps].tolist()
        X.append(seq_x)
    return X

def send_for_prediction(X):
    try:
        # Convert NumPy array to a list for JSON serialization
        if isinstance(X, np.ndarray):
            X_serializable = X.tolist()
        else:
            X_serializable = X
        
        response = requests.post(FASTAPI_URL, json={"data": X_serializable})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get predictions. Status code: {response.status_code}, Message: {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def main():
    st.title("ESP Anomaly Detection")
    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        st.write("Data preview:", df.head())

        process_button = st.button("Load and Process Data")
        if process_button:
            column_name = 'Ampere'  # Example column
            if column_name in df.columns:
                df_processed, scaler = preprocess_data(df, column_name)
                time_steps = 2
                X = create_sequences(df_processed[column_name].values, time_steps)
                predictions = send_for_prediction(X)
                if predictions:
                    df_processed = df_processed.iloc[time_steps:]
                    df_processed['Prediction'] = np.array(predictions['predictions'])

                    # Plotting
                    plt.figure(figsize=(10, 4))
                    plt.plot(df_processed['Date'], scaler.inverse_transform(df_processed[[column_name]]), label='Actual')
                    plt.plot(df_processed['Date'], scaler.inverse_transform(np.array(df_processed['Prediction']).reshape(-1, 1)), label='Predicted', color='red')
                    plt.legend()
                    plt.title('Predictions vs Actual Data')
                    plt.xlabel('Date')
                    plt.ylabel(column_name)
                    st.pyplot(plt)
            else:
                st.error(f"The selected column '{column_name}' does not exist in the uploaded file.")

if __name__ == '__main__':
    main()
