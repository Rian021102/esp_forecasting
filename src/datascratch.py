import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_data(df, x_column):

#plot based on the column name provided where y_column DATE data
    """
    Plot the data from the DataFrame based on the specified x_column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to plot.
    x_column (str): The name of the column to use for the x-axis.
    """
    if x_column not in df.columns:
        print(f"Column '{x_column}' not found in DataFrame.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df[x_column], df['DATE'], marker='o')  # Assuming 'DATE' is a column in the DataFrame
    plt.title(f"Plot of DATE vs {x_column}")
    plt.xlabel(x_column)
    plt.ylabel('DATE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    path="/home/rianr/pypro/myvenv/esp_forecasting/data/wells_data_final.csv" 
    df = load_data(path)
    print(df.head())  # Display the first few rows of the DataFrame

    #plot AMPERAGE for each WELL_ID column
    wells=df['PUMP'].unique()
    for well in wells:
        well_data = df[df['WELL_ID'] == well]
        plot_data(well_data, 'AMPERAGE')

if __name__ == "__main__":
    main()
