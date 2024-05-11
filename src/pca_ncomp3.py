import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def read_data(path):
    df = pd.read_csv(path)
    df.replace('No Data', np.nan, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.dropna(inplace=True)
    numeric_columns = ['Frequency', 'Voltage', 'Ampere', 'Pressure_Discharge', 'Pressure_Intake', 'Temp_Intake', 'Temp_Motor', 'Vibration_X', 'Vibration_Y']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    return df

def scale_data(df, num_columns):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[num_columns])
    return df_scaled

def apply_pca(df_scaled, df, numeric_columns):
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(df_scaled)
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)

    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'], index=df.index)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df_pca[['PC1', 'PC2']])
    df_pca['Cluster'] = kmeans.labels_
    df_combined = pd.concat([df, df_pca], axis=1)

    plt.figure(figsize=(15, 10))
    for i, combination in enumerate([(0, 1), (0, 2), (1, 2)], 1):
        ax = plt.subplot(2, 2, i)
        colors = ['blue', 'red', 'green']
        cluster_labels = list(range(3))
        for color, label in zip(colors, cluster_labels):
            indices = df_pca['Cluster'] == label
            ax.scatter(df_pca.loc[indices, f'PC{combination[0]+1}'], df_pca.loc[indices, f'PC{combination[1]+1}'], c=color, label=f'Cluster {label}', alpha=0.5)

        vectors = pca.components_.T * np.sqrt(pca.explained_variance_)
        for j, col in enumerate(numeric_columns):
            ax.arrow(0, 0, vectors[j, combination[0]], vectors[j, combination[1]], color='black', alpha=0.9, width=0.01)
            ax.text(vectors[j, combination[0]] * 1.2, vectors[j, combination[1]] * 1.2, col, color='black', ha='center', va='center')

        ax.set_xlabel(f'PC{combination[0]+1} ({round(pca.explained_variance_ratio_[combination[0]]*100, 1)}% expl.var)')
        ax.set_ylabel(f'PC{combination[1]+1} ({round(pca.explained_variance_ratio_[combination[1]]*100, 1)}% expl.var)')
        ax.set_title(f'Biplot with PCA - PC{combination[0]+1} vs PC{combination[1]+1}')
        ax.grid(True)
        ax.axis('equal')

    plt.legend()
    plt.tight_layout()
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

def main():
    path = '/Users/rianrachmanto/pypro/data/data_esp_edit02.csv'
    df = read_data(path)
    numeric_columns = ['Frequency', 'Voltage', 'Ampere', 'Pressure_Discharge', 'Pressure_Intake', 'Temp_Intake', 'Temp_Motor', 'Vibration_X', 'Vibration_Y']
    df_scaled = scale_data(df, numeric_columns)
    df_combined = apply_pca(df_scaled, df, numeric_columns)
    print(df_combined.head())
    cluster_means = cluster_averages(df_combined)
    print(cluster_means)  # Print the mean values by cluster
    plot_cluster_averages(cluster_means)

if __name__ == '__main__':
    main()
