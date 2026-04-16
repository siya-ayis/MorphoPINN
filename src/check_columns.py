import pandas as pd

# Load just the first 5 rows to inspect headers
try:
    # Try reading normally
    df = pd.read_csv('data/raw/ncei_microplastics.csv', nrows=5)
    print("--- HEADERS FOUND ---")
    print(df.columns.tolist())
    print("\n--- FIRST ROW ---")
    print(df.iloc[0])
except Exception as e:
    print(f"Error reading file: {e}")

    # Try reading with 'comment' support (common for NOAA files)
    print("\n--- ATTEMPTING TO SKIP METADATA ---")
    df = pd.read_csv('data/raw/ncei_microplastics.csv', nrows=5, comment='#')
    print(df.columns.tolist())