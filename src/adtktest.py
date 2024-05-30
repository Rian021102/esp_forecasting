import pandas as pd
import matplotlib.pyplot as plt
from adtk.data import validate_series
from adtk.detector import QuantileAD
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

def clean_data(df_feat):
    # Drop the missing values
    df_feat.dropna(inplace=True)
    return df_feat

def train_adtk_quantile_detector(df_feat, low_quantile, high_quantile):
    # Validate the series
    s = validate_series(df_feat)

    # Create a Quantile Anomaly Detector
    quantile_ad = QuantileAD(low=low_quantile, high=high_quantile)
    
    # Fit the detector (not necessary for QuantileAD, but for consistency in pipeline)
    quantile_ad.fit(s)

    # Detect anomalies
    anomalies = quantile_ad.detect(s)
    logging.info(f'Number of outliers: {anomalies.sum()}')
    outliers = df_feat[anomalies[feat_name]]
    logging.info(outliers)
    return quantile_ad, anomalies, df_feat

def plot_data_with_outliers(df_feat, feat_name, anomalies):
    plt.figure(figsize=(12, 6))
    plt.plot(df_feat.index, df_feat[feat_name], label='Data', zorder=1)  # Ensure data is plotted beneath outliers
    # Plot outliers
    outliers = df_feat.loc[anomalies[feat_name], feat_name]  # Correct way to access the outliers
    plt.scatter(outliers.index, outliers, color='red', label='Outliers', zorder=2)
    plt.xlabel('Date')
    plt.ylabel(feat_name)
    plt.title(f'{feat_name} over Time with Outliers Marked')
    plt.legend()
    plt.show()

def main(path, well_name, feat_name, low_quantile, high_quantile):
    df = load_data(path)
    df_feat = select_feat(df, well_name, feat_name)
    df_feat = clean_data(df_feat)
    quantile_ad, anomalies, df_feat = train_adtk_quantile_detector(df_feat, low_quantile, high_quantile)
    plot_data_with_outliers(df_feat, feat_name, anomalies)

if __name__ == '__main__':
    path = '/Users/rianrachmanto/pypro/data/esp_new_02.csv'
    well_name = 'YWB-15'
    feat_name = 'Ampere'
    low_quantile = 0.15
    high_quantile = 0.90
    main(path, well_name, feat_name, low_quantile, high_quantile)
