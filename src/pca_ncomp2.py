import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#import knn imputer
from sklearn.impute import KNNImputer

def read_data(path):
    df = pd.read_csv(path)
    df.replace('No Data', np.nan, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    print(df.info())
    df.dropna(inplace=True)
    numeric_columns = ['Frequency', 'Voltage', 'Ampere', 'Pressure_Discharge', 'Pressure_Intake', 'Temp_Intake', 'Temp_Motor', 'Vibration_X', 'Vibration_Y']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    return df

def scale_data(df, num_columns):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[num_columns])
    return df_scaled

def apply_pca(df_scaled, df, numeric_columns):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_scaled)
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)

    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=df.index)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df_pca)
    df_pca['Cluster'] = kmeans.labels_
    df_combined = pd.concat([df, df_pca], axis=1)

    # Create a biplot
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    # Scatter plot for clusters
    colors = ['blue', 'red', 'green']  # Define cluster colors
    cluster_labels = list(range(0, 3))
    for color, label in zip(colors, cluster_labels):
        indices = df_pca['Cluster'] == label
        plt.scatter(df_pca.loc[indices, 'PC1'], df_pca.loc[indices, 'PC2'], c=color, label=f'Cluster {label}', alpha=0.5)
    
    # Plot arrows and labels for each feature
    vectors = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, col in enumerate(numeric_columns):
        plt.arrow(0, 0, vectors[i, 0], vectors[i, 1], color='black', alpha=0.9, width=0.01)
        plt.text(vectors[i, 0] * 1.2, vectors[i, 1] * 1.2, col, color='black', ha='center', va='center')

    plt.xlabel('PC1 ({}% expl.var)'.format(round(pca.explained_variance_ratio_[0]*100, 1)))
    plt.ylabel('PC2 ({}% expl.var)'.format(round(pca.explained_variance_ratio_[1]*100, 1)))
    plt.title('Biplot with PCA')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

    return df_combined

def cluster_averages(df_combined):
    cluster_means = df_combined.groupby('Cluster').mean()
    return cluster_means

def plot_cluster_averages(cluster_means):
    
    cluster_means_transposed = cluster_means.transpose()
    cluster_means_transposed.plot(kind='bar', figsize=(14, 7))
    plt.title('Average Values of Features by Cluster')
    plt.xlabel('Features')
    plt.ylabel('Average Value')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()

def average_values(df_combined):
    # Group by 'Cluster' only and calculate the mean of 'Voltage' and 'Ampere'
    electrical_features = df_combined.groupby('Cluster')[['Frequency','Voltage', 'Ampere']].mean()

    # Print the resulting DataFrame
    print(electrical_features)

    # Transpose the DataFrame so that 'Voltage' and 'Ampere' are on the x-axis
    transposed_features = electrical_features.T

    # Plotting the bar chart
    ax = transposed_features.plot(kind='bar', figsize=(10, 6), width=0.8)

    # Setting plot titles and labels
    plt.title('Average Voltage and Ampere for each Cluster')
    plt.ylabel('Average Values')
    plt.xticks(rotation=0)  # Rotate x-axis labels to horizontal
    plt.legend(title='Cluster')
    plt.grid(True, linestyle='--', alpha=0.6)  # Adding a grid for better readability
    plt.show()




def main():
    path = '/Users/rianrachmanto/pypro/data/data_esp_edit02.csv'
    df = read_data(path)
    numeric_columns = ['Frequency', 'Voltage', 'Ampere', 'Pressure_Discharge', 'Pressure_Intake', 'Temp_Intake', 'Temp_Motor', 'Vibration_X', 'Vibration_Y']
    df_scaled = scale_data(df, numeric_columns)
    df_combined = apply_pca(df_scaled, df, numeric_columns)
    print(df_combined.head())
    #cluster_means = cluster_averages(df_combined)
    #print(cluster_means)  # Print the mean values by cluster
    #plot_cluster_averages(cluster_means)
    average_values(df_combined)
    


if __name__ == '__main__':
    main()
