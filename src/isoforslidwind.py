import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path):
    try:
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def select_feat(df, well_name, feat_name):
    df_feat = df[df['Well'] == well_name][['Well', 'Date', feat_name]]
    df_feat.set_index('Date', inplace=True)
    df_feat.drop(columns='Well', inplace=True)
    return df_feat

def clean_data(df_feat, feat_name):
    imputer = KNNImputer(n_neighbors=5)
    df_feat[feat_name] = imputer.fit_transform(df_feat[feat_name].values.reshape(-1, 1))
    return df_feat

def train_isolation_forest(df_feat, feat_name, contamination, window_size):
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predicted_full = pd.Series(index=df_feat.index, dtype=bool)
    
    # Iterate over the sliding window
    for start in range(0, len(df_feat), window_size):
        end = min(start + window_size, len(df_feat))
        data_slice = df_feat[feat_name].iloc[start:end].values.reshape(-1, 1)
        iso_forest.fit(data_slice)
        predicted_slice = iso_forest.predict(data_slice)
        predicted_full.iloc[start:end] = predicted_slice == -1
    
    logging.info(f'Number of outliers: {predicted_full.sum()}')
    return iso_forest, predicted_full, df_feat

#make function to plot the data series only
def plot_data(df_feat, feat_name):
    plt.figure(figsize=(12, 6))
    plt.plot(df_feat.index, df_feat[feat_name])
    plt.xlabel('Date')
    plt.ylabel(feat_name)
    plt.title(f'{feat_name} over Time')
    plt.show()

def plot_data_with_outliers(df_feat, feat_name, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(df_feat.index, df_feat[feat_name], label='Data', zorder=1)
    outliers = df_feat.loc[predicted, feat_name]
    plt.scatter(outliers.index, outliers, color='red', label='Outliers', zorder=2)
    plt.xlabel('Date')
    plt.ylabel(feat_name)
    plt.title(f'{feat_name} over Time with Outliers Marked')
    plt.text(0.5, 0.5, 'Isolation Forest', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=16)
    plt.legend()
    plt.show()

def main(path, well_name, feat_name, contamination, window_size):
    df = load_data(path)
    df_feat = select_feat(df, well_name, feat_name)
    df_feat = clean_data(df_feat, feat_name)
    plot_data(df_feat, feat_name)
    iso_forest, predicted, df_feat = train_isolation_forest(df_feat, feat_name, contamination, window_size)
    plot_data_with_outliers(df_feat, feat_name, predicted)
    
if __name__ == '__main__':
    path = '/Users/rianrachmanto/pypro/data/esp_new_02.csv'
    well_name = 'YWB-15'
    feat_name = 'Ampere'
    contamination =0.05
    window_size = 7  # Define window size here
    main(path, well_name, feat_name, contamination, window_size)
