import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.copod import COPOD

def load_data(path):
    df = pd.read_csv(path)
    print(df.info())
    # Convert the date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    print(df.head())
    print(df.info())
    return df

def select_feat(df, well_name, feat_name):
    # Filter the DataFrame for the given well name
    df_feat = df[df['Well'] == well_name][['Well', 'Date', feat_name]]
    print(df_feat.head())
    # Set the date as the index
    df_feat.set_index('Date', inplace=True)
    # Drop the well column
    df_feat.drop(columns='Well', inplace=True)
    return df_feat

def clean_data(df_feat, feat_name):
    # Drop the missing values
    df_feat.dropna(inplace=True)
    print('length of data after cleaning:', len(df_feat))
    return df_feat

def train_copod(df_feat, feat_name, contamination):
    # Create a COPOD model
    copod = COPOD(contamination=contamination)
    # Fit the model
    # Reshape the data to a 2D array
    data_2d = df_feat[feat_name].values.reshape(-1, 1)
    copod.fit(data_2d)
    # Get the prediction labels of the training data
    # Also, use the reshaped data for predictions
    predicted = pd.Series(copod.predict(data_2d), index=df_feat.index)
    print('Number of outliers:', predicted.sum())
    outliers = predicted[predicted == 1]
    outliers = df_feat.loc[outliers.index]
    print(outliers)
    return copod, predicted, df_feat

def plot_data_with_outliers(df_feat, feat_name, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(df_feat.index, df_feat[feat_name], label='Data')
    #plot predicted along with df_feat
    plt.plot(df_feat.index, df_feat[feat_name], color='green', label='Data')
    # Plot outliers
    outliers = df_feat.loc[predicted == 1, feat_name]
    plt.scatter(outliers.index, outliers, color='red', label='Outliers')
    #add text that says anomaly detection using COPOD on the left
    plt.text(df_feat.index[0], df_feat[feat_name].max(), 'Anomaly Detection using COPOD', fontsize=12, color='red')
    plt.xlabel('Date')
    plt.ylabel(feat_name)
    plt.title(f'{feat_name} over Time with Outliers Marked')
    plt.legend()
    plt.show()

def main():
    path = '/Users/rianrachmanto/pypro/data/esp_new_02.csv'
    df = load_data(path)
    well_name = 'YWB-15'
    feat_name = 'Ampere'
    df_feat = select_feat(df, well_name, feat_name)
    df_feat = clean_data(df_feat, feat_name)
    contamination = 0.2
    copod, predicted, df_feat = train_copod(df_feat, feat_name, contamination)
    print('length of predicted:', len(predicted))
    plot_data_with_outliers(df_feat, feat_name, predicted)

if __name__ == '__main__':
    main()
