import pandas as pd
import sys

def analyze():
    print("Loading NCEI Microplastics Dataset...")
    try:
        df = pd.read_csv('c:/Users/Admin/Desktop/MorphoGraph/data/raw/ncei_microplastics.csv')
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    print('\n=== DATAFRAME INFO ===')
    df.info()
    
    print('\n=== MISSING VALUES (Count > 0) ===')
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    print('\n=== CATEGORICAL COLUMN SNAPSHOT ===')
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        print(f'\n-> Column: {col}')
        print(df[col].value_counts().head(10))

if __name__ == "__main__":
    analyze()
