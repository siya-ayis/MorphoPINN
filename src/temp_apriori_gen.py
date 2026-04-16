import pandas as pd
import os

print("[Status] Generating transactional states for Apriori...")
for km in [15, 50, 100, 150, 200]:
    master_path = f'data/master/morpho_graph_master_{km}km.csv'
    if not os.path.exists(master_path):
        continue
    
    master = pd.read_csv(master_path)
    tx = master[['Morphology', 'Node_Cluster_ID', 'Is_Interpolated']].copy()
    tx['Node_Cluster_ID'] = 'Region_' + tx['Node_Cluster_ID'].astype(str)
    tx['Is_Interpolated'] = 'Interp_' + tx['Is_Interpolated'].astype(str)
    
    save_path = f'data/processed/apriori_transactions_{km}km.csv'
    tx.to_csv(save_path, index=False, header=False)
    print(f" -> Generated {save_path}")
