import pandas as pd
import re

# Dictionary of well identifiers
dict_data_tab0 = {
    'BC1': 'BC1', 'BS3': 'BS3', 'YNA7': 'YNA7', 'YNB8': 'YNB8', 'YNB17': 'YNB17', 'YNB28': 'YNB28',
    'YNB24': 'YNB24', 'YNC10': 'YNC10', 'YNC12': 'YNC12', 'YCA5': 'YCA5', 'YCA10': 'YCA10',
    'YCA7': 'YCA7', 'YCA8': 'YCA8', 'YNB29': 'YNB29', 'YNB30': 'YNB30', 'YCB4': 'YCB4',
    'YNB23': 'YNB23', 'YNB19': 'YNB19', 'YWB15': 'YWB15', 'YWA20': 'YWA20', 'YWA21': 'YWA21',
    'YWA23': 'YWA23', 'YWB22': 'YWB22', 'YWB19': 'YWB19', 'YCA11': 'YCA11', 'YWB14': 'YWB14',
    'YWB17': 'YWB17', 'YWB12': 'YWB12'
}

def process_sheet(path, sheet_name, value_name):
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    df = pd.melt(df, id_vars=['Date'], var_name='Well', value_name=value_name)
    # Regex to extract well name
    pattern = '|'.join(re.escape(key) for key in dict_data_tab0.keys())
    df['Well_ID'] = df['Well'].str.extract(f'({pattern})', expand=False)
    df.drop(columns=['Well'], inplace=True)
    return df[['Well_ID', 'Date', value_name]]

# Load and process each sheet
file_path = '/Users/rianrachmanto/pypro/ESP_WELL_YKN.xlsx'
sheets = ['Frequency', 'Voltage', 'Ampere', 'Pressure Discharge', 'Pressure Intake',
          'Temp Intake', 'Temp Motor', 'Vibration X', 'Vibration Y']

data_frames = [process_sheet(file_path, sheet, sheet.replace(' ', '_')) for sheet in sheets]
data_esp = data_frames[0]
for df in data_frames[1:]:
    data_esp = pd.merge(data_esp, df, on=['Well_ID', 'Date'], how='outer')

# Rename columns replacing spaces with underscores
data_esp.columns = data_esp.columns.str.replace(' ', '_')
print(data_esp.head())

# Save to CSV
data_esp.to_csv('/Users/rianrachmanto/pypro/data/data_esp.csv', index=False)
print("CSV file has been saved.")
