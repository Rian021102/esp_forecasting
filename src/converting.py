import pandas as pd
import numpy as np

df=pd.read_excel('/Users/rianrachmanto/pypro/data/data_esp.xlsx')
print(df.head())
#replace row contain Bad, Tag not found and 0 with NaN
df.replace(['Bad', 'Tag not found', 0], np.nan, inplace=True)
print(df.head())

#save to csv
df.to_csv('/Users/rianrachmanto/pypro/data/data_esp_edit02.csv', index=False)
