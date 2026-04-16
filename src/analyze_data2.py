import pandas as pd

def profile_data():
    df = pd.read_csv('c:/Users/Admin/Desktop/MorphoGraph/data/raw/ncei_microplastics.csv')
    with open('data_profile.txt', 'w', encoding='utf-8') as f:
        f.write("--- CORE SHAPE ---\n")
        f.write(f"Rows: {len(df)}, Cols: {len(df.columns)}\n\n")
        f.write("--- COLUMNS & MISSING ---\n")
        for col in df.columns:
            missing = df[col].isnull().sum()
            f.write(f"{col} | Type: {df[col].dtype} | Missing: {missing} ({(missing/len(df))*100:.1f}%)\n")
        
        f.write("\n--- SAMPLING METHOD UNIQUE VALUES ---\n")
        if 'Sampling Method' in df.columns:
            f.write(df['Sampling Method'].value_counts(dropna=False).head(15).to_string())
            
        f.write("\n\n--- OTHER IMPORTANT CATEGORICALS ---\n")
        for col in ['Measurement Type', 'Shape', 'Color', 'Polymer']:
            if col in df.columns:
                f.write(f"\n{col}:\n{df[col].value_counts(dropna=False).head(5).to_string()}\n")

if __name__ == '__main__':
    profile_data()
