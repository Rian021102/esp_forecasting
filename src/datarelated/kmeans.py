import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def read_data(path):
    df = pd.read_csv(path)
    df.replace('No Data', np.nan, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date parsing is needed for your analysis
    return df

def clean_data(df, columns):
    imputer = SimpleImputer(strategy='median')
    # Impute numeric columns
    df_cleaned = df.copy()  # Create a copy to preserve the original dataframe
    df_cleaned[columns] = imputer.fit_transform(df[columns])
    # Standardize numeric columns
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cleaned[columns])
    return df_cleaned, df_scaled  # Return both cleaned and scaled data

def elbow_method(df_scaled):
    distortions = []
    K = range(1, 11)

    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=5)
        kmeanModel.fit(df_scaled)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(16, 8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.show()

def kmeans_clustering(df_scaled, k,df_cleaned):
    kmeans = KMeans(n_clusters=k, random_state=3)
    kmeans.fit(df_scaled)
    df_cleaned['Cluster'] = kmeans.labels_
    print(df_cleaned)



def main():
    path = '/Users/rianrachmanto/pypro/data/data_esp_edit02.csv'
    df = read_data(path)
    numeric_columns = ['Frequency', 'Voltage', 'Ampere', 'Pressure_Discharge', 'Pressure_Intake', 'Temp_Intake', 'Temp_Motor', 'Vibration_X', 'Vibration_Y']
    df_cleaned, df_scaled = clean_data(df, numeric_columns)
    elbow_method(df_scaled)
    k = 3
    kmeans_clustering(df_scaled, k, df_cleaned)

if __name__ == '__main__':
    main()
