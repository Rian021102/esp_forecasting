import pandas as pd
import numpy as np
#import standard scaler
from sklearn.preprocessing import StandardScaler
#import PCA
from sklearn.decomposition import PCA
#import KMeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def read_data(path):
    df = pd.read_csv(path)
    df.replace('No Data', np.nan, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date parsing is needed for your analysis
    print(df.shape)
    #drop na
    df.dropna(inplace=True)
    print(df.shape)
    #convert Frequency, Voltage, Ampere, Pressure_Discharge, Pressure_Intake, Temp_Intake, Temp_Motor, Vibration_X, Vibration_Y to numeric
    numeric_columns = ['Frequency', 'Voltage', 'Ampere', 'Pressure_Discharge', 'Pressure_Intake', 'Temp_Intake', 'Temp_Motor', 'Vibration_X', 'Vibration_Y']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    print(df.isna().sum())
    return df

def scale_data(df, num_columns):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[num_columns])
    return df_scaled

def apply_pca(df_scaled, df):
    pca = PCA(n_components=2)  # Adjust the number of components as needed
    principal_components = pca.fit_transform(df_scaled)
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=df.index)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3)  # Adjust the number of clusters as needed
    kmeans.fit(df_pca)
    df_pca['Cluster'] = kmeans.labels_
    
    # Combine the PCA results and cluster labels with the original data
    df_combined = pd.concat([df, df_pca], axis=1)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    colors = ['red', 'green', 'blue']
    for i in range(3):
        plt.scatter(df_combined.loc[df_combined['Cluster'] == i, 'PC1'],
                    df_combined.loc[df_combined['Cluster'] == i, 'PC2'],
                    s=50, c=colors[i], label=f'Cluster {i}')
    plt.title('Clusters of ESP Data on PCA Components')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True)
    plt.show()

    return df_combined

def main():
    path = '/Users/rianrachmanto/pypro/data/data_esp_edit02.csv'
    df = read_data(path)
    numeric_columns = ['Frequency', 'Voltage', 'Ampere', 'Pressure_Discharge', 'Pressure_Intake', 'Temp_Intake', 'Temp_Motor', 'Vibration_X', 'Vibration_Y']
    df_scaled = scale_data(df, numeric_columns)
    df_final = apply_pca(df_scaled, df)
    print(df_final.head())

if __name__ == '__main__':
    main()
