import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
import logging
from matplotlib.animation import FuncAnimation

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

def setup_animation(df_feat, feat_name, contamination):
    fig, ax = plt.subplots(figsize=(12, 6))
    anomalies = pd.Series(data=False, index=df_feat.index)
    iso_forest = IsolationForest(contamination=contamination, random_state=42)

    def animate(i):
        if i < 7:
            return  # Need at least one full window to start
        
        end = min(i + 1, len(df_feat))
        data_window = df_feat.iloc[max(0, end-7):end]
        iso_forest.fit(data_window[[feat_name]])
        predicted = iso_forest.predict(data_window[[feat_name]]) == -1
        
        # Update the anomalies Series
        anomalies[data_window.index] = predicted

        ax.clear()
        ax.plot(df_feat.index[:end], df_feat[feat_name][:end], label='Data', zorder=1)
        outlier_points = df_feat[anomalies][:end]
        ax.scatter(outlier_points.index, outlier_points[feat_name], color='red', label='Outliers', zorder=2)
        ax.set_xlabel('Date')
        ax.set_ylabel(feat_name)
        ax.set_title(f'{feat_name} over Time with Outliers Marked')
        ax.legend()

    ani = FuncAnimation(fig, animate, frames=len(df_feat), repeat=False)
    plt.show()

def main(path, well_name, feat_name, contamination):
    df = load_data(path)
    df_feat = select_feat(df, well_name, feat_name)
    df_feat = clean_data(df_feat, feat_name)
    setup_animation(df_feat, feat_name, contamination)

if __name__ == '__main__':
    path = '/Users/rianrachmanto/pypro/data/esp_new_02.csv'
    well_name = 'YWB-15'
    feat_name = 'Volt'
    contamination = 0.05
    main(path, well_name, feat_name, contamination)
