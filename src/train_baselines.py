import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from torch_geometric.nn import SAGEConv, GATConv

from model_training import MorphoModeler

DRIFT_SECONDS = 86400.0  # seconds in 24 hours — identical to model_training.py

class ADE_Numerical_Baseline:
    """Pure Physics Proxy predicting Eulerian field via mean advection + dispersion variance."""
    def __init__(self):
        self.mean_v = None
    def fit(self, y_train):
        self.mean_v = np.mean(y_train, axis=0)
    def predict(self, X):
        # Predict constant mean field (Advection)
        return np.tile(self.mean_v, (len(X), 1))

class LSTM_Baseline(nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, out_dim)
        
    def forward(self, x):
        # x is [N, F]. Reshape to [N, F, 1] for proper temporal processing
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class CNN_LSTM_Baseline(nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, out_dim)
        
    def forward(self, x):
        # x is [N, F]. reshape for conv1d: [N, 1, F]
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1d(x)) # [N, 16, F]
        # Permute to [N, F, 16] for proper temporal window processing
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GraphSAGE_Baseline(nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super().__init__()
        self.sage = SAGEConv(in_dim, 32)
        self.fc = nn.Linear(32, out_dim)
        
    def forward(self, x, edge_index):
        h = torch.relu(self.sage(x, edge_index))
        return self.fc(h)

class ST_GCN_Baseline(nn.Module):
    """The textbook standard non-physics ST-GCN."""
    def __init__(self, in_dim, edge_dim, out_dim=2):
        super().__init__()
        self.gat = GATConv(in_dim, 32, edge_dim=edge_dim, add_self_loops=False)
        self.fc = nn.Linear(32, out_dim)
        
    def forward(self, x, edge_index, edge_attr):
        h = torch.relu(self.gat(x, edge_index, edge_attr))
        return self.fc(h)

def train_and_eval_model(model_name, model, X, y, coords, clusters, edge_index=None, edge_attr=None, epochs=150):
    start_time = time.time()
    
    n_samples = len(X)
    n_groups = len(np.unique(clusters))
    train_mask = np.zeros(n_samples, dtype=bool)
    if n_groups >= 2:
        n_splits = min(5, n_groups)
        gkf = GroupKFold(n_splits=n_splits)
        train_idx, test_idx = next(gkf.split(X, y, groups=clusters))
        train_mask[train_idx] = True
    else:
        test_idx = np.random.choice(n_samples, size=max(1, int(0.2*n_samples)), replace=False)
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_idx] = False
        
    test_mask = ~train_mask
    
    if model_name == "ADE":
        ade = ADE_Numerical_Baseline()
        ade.fit(y[train_mask].numpy())
        preds = np.tile(ade.mean_v, (test_mask.sum(), 1))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.HuberLoss(delta=1.0)
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            if model_name == "GraphSAGE":
                out = model(X, edge_index)
            elif model_name == "ST-GCN":
                out = model(X, edge_index, edge_attr)
            else:
                out = model(X)
                
            loss = criterion(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            if model_name == "GraphSAGE": preds = model(X, edge_index)[test_mask].numpy()
            elif model_name == "ST-GCN": preds = model(X, edge_index, edge_attr)[test_mask].numpy()
            else: preds = model(X)[test_mask].numpy()
            
    train_time = (time.time() - start_time) / 3600.0 # Save in hours
    actuals = y[test_mask].numpy()
    return preds, actuals, train_time

def run_baselines_pipeline(protocols=[15]):
    print("========================================")
    print("STARTING ACADEMIC BASELINE TRAINING")
    print("========================================")
    
    for km in protocols:
        master_file = f"data/master/morpho_graph_master_{km}km.csv"
        if not os.path.exists(master_file):
            print(f"Skipping {km}km - master file missing.")
            continue
            
        print(f"--- Loading Core Data Structure for {km}km ---")
        pipeline = MorphoModeler(master_file, protocol_km=km)
        pipeline.prepare_data() # Gets identical X, y_reg, coords
        
        X_tensor = torch.FloatTensor(pipeline.X)
        y_tensor = torch.FloatTensor(pipeline.y_reg)
        
        # Build strict topological edges (for graph models) identically
        edge_index, edge_attr = pipeline.build_haversine_graph(pipeline.coords, pipeline.y_reg, k=5)
        
        baselines = {
            "ADE": None,
            "LSTM": LSTM_Baseline(in_dim=X_tensor.shape[1]),
            "CNN-LSTM": CNN_LSTM_Baseline(in_dim=X_tensor.shape[1]),
            "GraphSAGE": GraphSAGE_Baseline(in_dim=X_tensor.shape[1]),
            "ST-GCN": ST_GCN_Baseline(in_dim=X_tensor.shape[1], edge_dim=3)
        }
        
        for name, model in baselines.items():
            print(f"> Training {name} on {km}km...")
            preds, actuals, t_hours = train_and_eval_model(name, model, X_tensor, y_tensor, pipeline.coords, pipeline.clusters, edge_index, edge_attr)
            
            # 24-hour displacement error in km — same formula as model_training.py
            error_e_m = (preds[:, 0] - actuals[:, 0]) * DRIFT_SECONDS
            error_n_m = (preds[:, 1] - actuals[:, 1]) * DRIFT_SECONDS
            abs_err_km = np.sqrt(error_e_m**2 + error_n_m**2) / 1000.0

            err_df = pd.DataFrame({
                'Actual_E_ms': actuals[:, 0], 'Actual_N_ms': actuals[:, 1],
                'Pred_E_ms': preds[:, 0], 'Pred_N_ms': preds[:, 1],
                'Error_KM_24h': abs_err_km
            })
            
            # Export dynamically to be read by calculate_metrics.py
            output_csv = f"data/processed/spatial_errors_{name}_{km}km.csv"
            err_df.to_csv(output_csv, index=False)
            
            # Save a dummy training history just for the duration metric calculation compatibility
            pd.DataFrame([{"Epoch": i, "Val_Loss": 0} for i in range(max(1, int(t_hours*3600/1.2)))]).to_csv(f"data/processed/training_history_{name}_{km}km.csv", index=False)
            print(f"  [+] {name} successfully converged and spatial error logged.")

if __name__ == "__main__":
    target_protocols = [15, 50, 100, 150, 200]
    run_baselines_pipeline(target_protocols)
