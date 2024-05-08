import pandas as pd
import numpy as np
import re

# Load data
df = pd.read_csv('/Users/rianrachmanto/pypro/data/data_esp.csv')

# Function to check if a cell contains only words
def is_word(x):
    if isinstance(x, str):
        return bool(re.match('^[A-Za-z]+$', x))
    return False

# Applying the function to filter and print words
for col in df.columns:
    words = df[col].apply(is_word)
    unique_words = df[col][words].unique()
    print(f"Unique words in column {col}:")
    print(unique_words)
    print(f"Number of unique words: {len(unique_words)}")
    print(df[col][words].value_counts())
    print('-------------------')

#find rows that all the columns contain word bad and make new column called remark and fill it with 'bad'
df['remark'] = np.where(df.apply(lambda x: x.str.contains('bad').all(), axis=1), 'bad')
print(df[df['remark'] == 'bad'])
