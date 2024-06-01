import pandas as pd
import matplotlib.pyplot as plt
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import LevelShiftAD, PersistAD
from adtk.detector import OutlierDetector
from sklearn.neighbors import LocalOutlierFactor

df=pd.read_csv('/Users/rianrachmanto/pypro/data/esp_new_02.csv')
#convert to Date to datetime
df['Date'] = pd.to_datetime(df['Date'])
#filter to only show Well YWB-15 and column Ampere
df = df[df['Well'] == 'YWB-15']
df = df[['Date','Ampere']]
print(df.head())
#dropna 
df = df.dropna()
# set index to Date
df = df.set_index('Date')
print(df.head())
#validate series
s = validate_series(df)
#level shift anomaly detection
level_shift_ad = LevelShiftAD(c=0.5, side='both', window=10)
anomalies = level_shift_ad.fit_detect(s)
plot(s, anomaly=anomalies, anomaly_color='red')
plt.show()
# PersistAD anomaly detection
persist_ad = PersistAD(c=3.0, side='positive')
anomalies = persist_ad.fit_detect(s)
plot(s, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red')
plt.show()

outlier_detector = OutlierDetector(LocalOutlierFactor(contamination=0.05))
anomalies = outlier_detector.fit_detect(df)
plot(df, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3, curve_group='all')
plt.show()