import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
#import knn imputer
from sklearn.impute import KNNImputer

# write title and header
st.title('ESP Anomaly Detection')
st.header('Leveraging Isolation Forests for Anomaly Detection')

# data load button
st.subheader('Load Data')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())
    #convert the date to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    #drop Hours_Online column
    data.drop(columns='Hours_Online', inplace=True)
    #drop Gross_Rate column
    data.drop(columns='Gross_Rate', inplace=True)
    #convert numeric columns to float
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].astype(float)
    #make side bar with Well names taken from unique values in the 'Well' column
    well_name = st.sidebar.selectbox('Select Well Name', data['Well'].unique())
    # for each well selected perform imputaion for all numeric columns
    for col in numeric_cols:
        # Filter the DataFrame for the given well name
        df_feat = data[data['Well'] == well_name][['Well', 'Date', col]]
        # Set the date as the index
        df_feat.set_index('Date', inplace=True)
        # Drop the well column
        df_feat.drop(columns='Well', inplace=True)
        # Impute missing values using KNN imputer
        imputer = KNNImputer(n_neighbors=5)
        df_feat[col] = imputer.fit_transform(df_feat[col].values.reshape(-1, 1))
        iso_forest = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', 
                                     max_features=1.0, bootstrap=False, n_jobs=None, random_state=None, 
                                     verbose=0, warm_start=False)
        # Fit the model
        data_2d = df_feat[col].values.reshape(-1, 1)
        iso_forest.fit(data_2d)
        # Get the prediction labels of the training data
        predicted = pd.Series(iso_forest.predict(data_2d), index=df_feat.index)
        # Convert predictions from -1 (outlier) and 1 (inlier) to boolean
        predicted = (predicted == -1)
        #print the data with outliers as dataframe
        st.write(df_feat[predicted])
        #plot the data with outliers of each feature
        plt.figure(figsize=(12, 6))
        plt.plot(df_feat.index, df_feat[col], label='Data', zorder=1)
        plt.scatter(df_feat.index[predicted], df_feat[col][predicted], color='red', label='Outliers', zorder=2)
        plt.title(f'{well_name} - {col} with Outliers')
        plt.legend()
        st.pyplot(plt)

    
    