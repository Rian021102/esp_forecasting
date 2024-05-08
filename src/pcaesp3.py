import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This is needed for 3D plotting

def read_data(path):
    df = pd.read_csv(path)
    df.replace('No Data', np.nan, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def clean_data(df, columns):
    imputer = SimpleImputer(strategy='median')
    df[columns] = imputer.fit_transform(df[columns])
    return df

def apply_pca_and_cluster(df, columns):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[columns])
    pca = PCA(n_components=3)  # Using 3 principal components
    principal_components = pca.fit_transform(df_scaled)
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'], index=df.index)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3)  # Adjust the number of clusters as needed
    kmeans.fit(df_pca)
    df_pca['Cluster'] = kmeans.labels_
    
    # Combine the PCA results and cluster labels with the original data
    df_combined = pd.concat([df, df_pca], axis=1)
    
    # Plotting in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'green', 'blue']
    for i in range(3):
        ax.scatter(df_combined.loc[df_combined['Cluster'] == i, 'PC1'],
                   df_combined.loc[df_combined['Cluster'] == i, 'PC2'],
                   df_combined.loc[df_combined['Cluster'] == i, 'PC3'],
                   s=50, c=colors[i], label=f'Cluster {i}')
    ax.set_title('3D Clusters of ESP Data')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend()
    plt.show()

    return df_combined

def main():
    path = '/Users/rianrachmanto/pypro/data/data_esp_edit02.csv'
    numeric_columns = ['Frequency', 'Voltage', 'Ampere', 'Pressure_Discharge', 'Pressure_Intake', 'Temp_Intake', 'Temp_Motor', 'Vibration_X', 'Vibration_Y']
    df = read_data(path)
    df_cleaned = clean_data(df, numeric_columns)
    df_final = apply_pca_and_cluster(df_cleaned, numeric_columns)
    print(df_final.head())

if __name__ == '__main__':
    main()
