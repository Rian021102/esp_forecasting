import pandas as pd

# Load data
df = pd.read_csv('/Users/rianrachmanto/pypro/data/data_esp.csv')

#replace all columns with Tag not found and No Data with Bad
df = df.replace('Tag Not Found', 'Bad').replace('No Data', 'Bad')

# Specify columns to check for 'Bad'
columns_to_check = ['Frequency', 'Voltage', 'Ampere', 'Pressure_Discharge', 'Pressure_Intake', 
                    'Temp_Intake', 'Temp_Motor', 'Vibration_X', 'Vibration_Y']

# Check if 'Bad' exactly matches in the cell content, case insensitive
contains_bad = df[columns_to_check].applymap(lambda x: str(x).strip().lower() == 'bad')

# Determine the 'remark' for each row
def determine_remark(row):
    if row.all():
        return "All Bad"
    elif row.any():
        return "Not All Bad"
    else:
        return "No Bad"

df02=df.copy()

df02['remark'] = contains_bad.apply(determine_remark, axis=1)

# filter to only display remark 'No Bad"
df_no_bad = df02[df02['remark'] == 'No Bad']
print(df_no_bad)
#save df_no_bad to csv
df_no_bad.to_csv('/Users/rianrachmanto/pypro/data/data_esp_no_bad.csv', index=False)