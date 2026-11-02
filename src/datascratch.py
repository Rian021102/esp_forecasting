import os
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


def main():
    path="/home/rianr/pypro/myvenv/esp_forecasting/data/wells_data_final.csv" 
    df = load_data(path)
    print(df.head())  # Display the first few rows of the DataFrame

    df["DATE"] = pd.to_datetime(df["DATE"])

    out_dir = "/home/rianr/pypro/myvenv/esp_forecasting/image/wells"
    os.makedirs(out_dir, exist_ok=True)

    # One file per well; each pump installed in the well gets its own line
    for well_id, well in df.groupby("WELL_ID"):
        well = well.sort_values("DATE")

        fig, ax = plt.subplots(figsize=(12, 5))
        for pump, segment in well.groupby("PUMP", sort=False):
            ax.plot(segment["DATE"], segment["CURRENT"], label=pump)
        ax.set_title(f"CURRENT - {well_id}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Current")
        ax.legend(title="Pump")
        fig.tight_layout()

        fig.savefig(os.path.join(out_dir, f"{well_id}.png"), dpi=150)
        plt.close(fig)

    print(f"Saved {df['WELL_ID'].nunique()} plots to {out_dir}")


if __name__ == "__main__":
    main()
