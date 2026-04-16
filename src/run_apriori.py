import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import os
from scipy.stats import fisher_exact

def run_forensic_mining(protocol_km=50):
    print(f"[Status] Apriori Mining ({protocol_km}km)...")
    file_path = f'data/processed/apriori_transactions_{protocol_km}km.csv'
    if not os.path.exists(file_path): 
        print(f"File missing: {file_path}")
        return
    
    with open(file_path, 'r') as f:
        transactions = [line.strip().replace('"', '').split(',') for line in f.readlines()]

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Metric 2.2: Rules using true variables
    frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
    if frequent_itemsets.empty: return

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    valid_rules = []
    N = len(df)
    for idx, row in rules.iterrows():
        a = int(row['support'] * N)
        b = int((row['antecedent support'] - row['support']) * N)
        c = int((row['consequent support'] - row['support']) * N)
        d = int(N - (a + b + c))
        
        _, p_val = fisher_exact([[a, b], [c, d]], alternative='greater')
        
        # Metric 2.2: Reimposing the brutal Fisher's Exact Test natively (p < 0.05)
        if p_val < 0.05:
            row['fisher_p_value'] = p_val
            valid_rules.append(row)
    
    rules = pd.DataFrame(valid_rules)
    if rules.empty: return
        
    rules = rules.sort_values(by='lift', ascending=False)
    rules.to_csv(f'data/processed/forensic_rules_{protocol_km}km.csv', index=False)
    
    # Formulate Briefing Report
    with open('data/processed/PI_Briefing_Apriori.txt', 'w', encoding='utf-8') as f:
        f.write("--- GENUINE APRIORI SIGNATURES ---\n")
        
        def clean_set(s): return str(s).replace("frozenset({", "").replace("})", "").replace("'", "")
        rules['Rule'] = rules['antecedents'].apply(clean_set) + "  ➔  " + rules['consequents'].apply(clean_set)
        
        for i, row in rules.head(5).iterrows():
            f.write(f"[{row['Rule']}] -> Lift: {row['lift']:.2f} | Confidence: {row['confidence']*100:.1f}% | Fisher P-Value: {row['fisher_p_value']:.4e}\n")

if __name__ == "__main__":
    for km in [15, 50, 100, 150, 200]:
        run_forensic_mining(km)