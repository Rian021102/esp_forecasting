import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path):
    try:
        df = pd.read_csv(path)
        logging.info(df.info())
        # Convert the date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        logging.info(df.head())
        logging.info(df.info())
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def select_feat(df, well_name, feat_name):
    # Filter the DataFrame for the given well name
    df_feat = df[df['Well'] == well_name][['Well', 'Date', feat_name]]
    logging.info(df_feat.head())
    # Set the date as the index
    df_feat.set_index('Date', inplace=True)
    # Drop the well column
    df_feat.drop(columns='Well', inplace=True)
    return df_feat

def clean_data(df_feat, feat_name):
    # Drop the missing values
    df_feat.dropna(inplace=True)
    return df_feat

def train_isolation_forest(df_feat, feat_name, contamination):
    # Create an Isolation Forest model
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    # Fit the model
    data_2d = df_feat[feat_name].values.reshape(-1, 1)
    iso_forest.fit(data_2d)
    # Get the prediction labels of the training data
    predicted = pd.Series(iso_forest.predict(data_2d), index=df_feat.index)
    # Convert predictions from -1 (outlier) and 1 (inlier) to boolean
    predicted = (predicted == -1)
    logging.info(f'Number of outliers: {predicted.sum()}')
    outliers = df_feat[predicted]
    logging.info(outliers)
    return iso_forest, predicted, df_feat

def plot_data_with_outliers(df_feat, feat_name, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(df_feat.index, df_feat[feat_name], label='Data', zorder=1)  # Ensure data is plotted beneath outliers
    # Plot outliers
    outliers = df_feat.loc[predicted, feat_name]  # Correct way to access the outliers
    plt.scatter(outliers.index, outliers, color='red', label='Outliers', zorder=2)
    plt.xlabel('Date')
    plt.ylabel(feat_name)
    plt.title(f'{feat_name} over Time with Outliers Marked')
    # put text that says Isolation Forest
    plt.text(0.5, 0.5, 'Isolation Forest', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=16)
    plt.legend()
    plt.show()

def main(path, well_name, feat_name, contamination):
    df = load_data(path)
    df_feat = select_feat(df, well_name, feat_name)
    df_feat = clean_data(df_feat, feat_name)
    iso_forest, predicted, df_feat = train_isolation_forest(df_feat, feat_name, contamination)
    plot_data_with_outliers(df_feat, feat_name, predicted)

if __name__ == '__main__':
    path = '/Users/rianrachmanto/pypro/data/esp_new.csv'
    well_name = 'MHN-6'
    feat_name = 'Ampere'
    contamination = 0.05
    main(path, well_name, feat_name, contamination)
