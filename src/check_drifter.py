import pandas as pd

# Path to your drifter file
FILE_PATH = 'data/raw/gdp_drifter_hourly.csv'

try:
    # Read first 5 rows
    df = pd.read_csv(FILE_PATH, nrows=5)
    print("--- DRIFTER COLUMNS ---")
    print(df.columns.tolist())
    print("\n--- FIRST ROW ---")
    print(df.iloc[0])
except Exception as e:
    print(f"Error reading file: {e}")