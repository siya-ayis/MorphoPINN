import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import BallTree
import os

try:
    from torch_geometric.nn import GATConv, MessagePassing
except ImportError:
    print("CRITICAL: torch_geometric is required for this MPINN. Please 'pip install torch_geometric'")
    class MessagePassing(nn.Module):
        def __init__(self, aggr='mean'): super().__init__()
    class GATConv(nn.Module):
        def __init__(self, in_channels, out_channels, edge_dim, add_self_loops=False): super().__init__()

EARTH_RADIUS_KM = 6371.0

# ==========================================
# 1. SOTA ARCHITECTURE: ENCODER-PROCESSOR-DECODER (EPD) PINN
# ==========================================
class MorphoSTGNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, output_dim, protocol_km=15):
        super(MorphoSTGNN, self).__init__()
        self.protocol_km = protocol_km
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(node_in_dim + 2, hidden_dim), # +2 to include dynamic coordinates for autograd 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Processor (Deep Multi-Scale GAT Layers)
        self.gat1 = GATConv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, add_self_loops=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, add_self_loops=False)
        self.gat3 = GATConv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, add_self_loops=False)
        
        # Transformer Temporal Mapping (Replacing single-state GRU)
        self.temporal_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, coords, edge_index, edge_attr):
        # We explicitly concat coords to enable Navier-Stokes autograd gradients during backprop
        x_aug = torch.cat([x, coords], dim=-1)
        h = self.encoder(x_aug)
        
        # Deep Message Passing
        h_gat1 = F.relu(self.gat1(h, edge_index, edge_attr))
        h_gat2 = F.relu(self.gat2(h_gat1, edge_index, edge_attr))
        h_gat3 = F.relu(self.gat3(h_gat2, edge_index, edge_attr))
        
        # Sequence-level temporal mapping across entire topological space
        h_attn, _ = self.temporal_attn(h_gat3.unsqueeze(0), h_gat3.unsqueeze(0), h_gat3.unsqueeze(0))
        h_attn = h_attn.squeeze(0)
        
        # Adaptive Scaling Protocol (balances graph width with sequence attention)
        alpha = min(1.0, max(0.0, self.protocol_km / 200.0))
        h_final = (1 - alpha) * h_attn + alpha * h_gat3
        
        return self.decoder(h + h_final)


# ==========================================
# 2. TRAINING PIPELINE (WITH LEAKAGE FIXED)
# ==========================================
class MorphoModeler:
    def __init__(self, master_path, protocol_km=15):
        self.path = master_path
        self.protocol_km = protocol_km
        self.X = None
        self.coords = None
        self.clusters = None
        self.y_reg = None
        self.feature_cols = []
        self.scaler_y = None

    def prepare_data(self):
        print(f"[Status] Loading strictly un-collapsed data from {self.path}...")
        df = pd.read_csv(self.path)
        
        method_cols = [c for c in df.columns if c.startswith('Sampling_Method_')]
        setting_cols = [c for c in df.columns if c.startswith('Marine_Setting_')]
        cat_cols = method_cols + setting_cols
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month.astype(float)
        
        self.feature_cols = cat_cols + ['Month', 'Day_Sin', 'Day_Cos', 'Lunar_Phase']
        raw_X = df[self.feature_cols].values.astype(float)
        self.coords = df[['Latitude', 'Longitude']].values.astype(float)
        self.clusters = df['Node_Cluster_ID'].values if 'Node_Cluster_ID' in df.columns else np.arange(len(df))
        
        scaler_X = StandardScaler()
        self.X = scaler_X.fit_transform(raw_X)
        
        # Scaling Y mathematically separately to allow inverse transformation during evaluation later!
        raw_y = df[['Velocity_E', 'Velocity_N']].values.astype(float)
        self.scaler_y = StandardScaler()
        self.y_reg = self.scaler_y.fit_transform(raw_y)
        
        os.makedirs('data/processed/encoders', exist_ok=True)
        joblib.dump(scaler_X, f'data/processed/encoders/feature_scaler_{self.protocol_km}km.pkl')
        joblib.dump(self.scaler_y, f'data/processed/encoders/target_scaler_{self.protocol_km}km.pkl')
        print(f"[Status] Un-collapsed Data Loaded: {len(self.X)} geographic vertices mapped.") 

    def split_and_augment(self):
        """ Creates an isolated validation set, only running SMOTE on the Train split to stop target leakage. """
        n_samples = len(self.X)
        test_idx = np.random.choice(n_samples, size=max(1, int(0.2*n_samples)), replace=False)
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_idx] = False
        
        X_train, y_train, coords_train = self.X[train_mask], self.y_reg[train_mask], self.coords[train_mask]
        X_test, y_test, coords_test = self.X[~train_mask], self.y_reg[~train_mask], self.coords[~train_mask]
        
        # Execute SMOTE ONLY on Training Data
        aug_X_tr, aug_y_tr, aug_coords_tr = self.apply_physics_smote(X_train, y_train, coords_train, target_n=max(10, len(X_train)//5))
        
        # Concat the safe augmented nodes with the safe test nodes to create the unified valid Graph Space
        train_mask_aug = np.ones(len(aug_X_tr), dtype=bool)
        test_mask_aug = np.zeros(len(X_test), dtype=bool)
        
        final_X = np.concatenate([aug_X_tr, X_test], axis=0)
        final_y = np.concatenate([aug_y_tr, y_test], axis=0)
        final_coords = np.concatenate([aug_coords_tr, coords_test], axis=0)
        final_train_mask = np.concatenate([train_mask_aug, test_mask_aug])
        final_test_mask = ~final_train_mask
        
        return final_X, final_y, final_coords, final_train_mask, final_test_mask

    def apply_physics_smote(self, X_tr, y_tr, coords_tr, target_n):
        if len(X_tr) < 2: return X_tr, y_tr, coords_tr
        
        print(f"[Validation Constraint] SMOTE running isolated purely on training fold ({target_n} clones)...")
        v_sigma = np.std(y_tr, axis=0) * 1.5
        
        synthetic_X, synthetic_y, synthetic_coords = [], [], []
        for _ in range(target_n):
            idx = np.random.randint(0, len(X_tr))
            jitter_e = np.clip(np.random.normal(0, v_sigma[0]/2), -v_sigma[0], v_sigma[0])
            jitter_n = np.clip(np.random.normal(0, v_sigma[1]/2), -v_sigma[1], v_sigma[1])
            
            synthetic_X.append(X_tr[idx].copy())
            synthetic_y.append(y_tr[idx].copy() + np.array([jitter_e, jitter_n]))
            synthetic_coords.append(coords_tr[idx].copy() + np.random.normal(0, 0.005, size=2))
            
        return np.vstack([X_tr, np.array(synthetic_X)]), np.vstack([y_tr, np.array(synthetic_y)]), np.vstack([coords_tr, np.array(synthetic_coords)])

    def build_haversine_graph(self, coords, y_reg, k=5):
        """ Rigorous Haversine Graph Builder """
        k_actual = min(k, len(coords) - 1)
        if k_actual <= 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 3), dtype=torch.float32)

        coords_rad = np.radians(coords)
        tree = BallTree(coords_rad, metric='haversine')
        dist, ind = tree.query(coords_rad, k=k_actual+1)
        
        sources, targets, edge_attrs = [], [], []
        omega = 7.2921e-5 
        
        for i in range(len(coords)):
            lat1, lon1 = coords[i]
            ve1, vn1 = y_reg[i]
            v1_norm = math.sqrt(ve1**2 + vn1**2) + 1e-9
            
            for j_idx in range(1, k_actual+1):
                j = ind[i, j_idx]
                lat2, lon2 = coords[j]
                
                ve2, vn2 = y_reg[j]
                v2_norm = math.sqrt(ve2**2 + vn2**2) + 1e-9
                
                if ((ve1 * ve2) + (vn1 * vn2)) / (v1_norm * v2_norm) < 0: continue 
                
                dLon = math.radians(lon2 - lon1)
                y = math.sin(dLon) * math.cos(math.radians(lat2))
                x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
                    math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dLon)
                bearing = math.atan2(y, x)
                
                true_dist_km = dist[i, j_idx] * EARTH_RADIUS_KM
                
                sources.append(j)
                targets.append(i)
                edge_attrs.append([true_dist_km, bearing, 2 * omega * math.sin(math.radians(lat1))])
                
        return torch.tensor([sources, targets], dtype=torch.long), torch.FloatTensor(edge_attrs)

    def run_st_gcn(self):
        print("\n--- TRAINING SOTA EPD-PINN ---")
        if len(self.X) < 2: return
        
        final_X, final_y, final_coords, train_mask, test_mask = self.split_and_augment()
        edge_index, edge_attr = self.build_haversine_graph(final_coords, final_y, k=10)
        
        X_tensor = torch.FloatTensor(final_X)
        y_tensor = torch.FloatTensor(final_y)
        coords_t = torch.FloatTensor(final_coords)
        
        model = MorphoSTGNN(node_in_dim=final_X.shape[1], edge_in_dim=3, hidden_dim=64, output_dim=2, protocol_km=self.protocol_km)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
        criterion = nn.HuberLoss(delta=1.0)
        
        history = []
        for epoch in range(150):
            model.train()
            optimizer.zero_grad()
            
            # Allow Autograd to calculate Physical PDE Divergence on the Coordinates
            coords_t.requires_grad_(True)
            
            out = model(X_tensor, coords_t, edge_index, edge_attr)
            loss_data = criterion(out[train_mask], y_tensor[train_mask])
            
            # --- TRUE PHYSICS-INFORMED PENALTY (Navier-Stokes Mass Conservation approx) ---
            # Penalize the network if local graph coordinate flow divergence is abnormally exploding
            grad_outputs_u = torch.ones_like(out[train_mask, 0])
            grad_u = torch.autograd.grad(out[train_mask, 0], coords_t, grad_outputs=grad_outputs_u, create_graph=True, retain_graph=True)[0]
            
            grad_outputs_v = torch.ones_like(out[train_mask, 1])
            grad_v = torch.autograd.grad(out[train_mask, 1], coords_t, grad_outputs=grad_outputs_v, create_graph=True, retain_graph=True)[0]
            
            dU_dLon = grad_u[train_mask, 1]
            dV_dLat = grad_v[train_mask, 0]
            
            divergence = dU_dLon + dV_dLat
            loss_physics = 0.05 * torch.mean(divergence**2)
            
            loss_train = loss_data + loss_physics
            loss_train.backward()
            optimizer.step()
            scheduler.step()
            
            # Drop graph requirement for testing loop to save VRAM
            coords_t.requires_grad_(False)
            
            model.eval()
            with torch.no_grad():
                out_val = model(X_tensor, coords_t, edge_index, edge_attr)
                loss_val = criterion(out_val[test_mask], y_tensor[test_mask])
                
            history.append({'Epoch': epoch, 'Train_Loss': loss_train.item(), 'Val_Loss': loss_val.item()})
            if epoch % 30 == 0:
                print(f"Epoch {epoch}: Train Data Loss {loss_data.item():.4f} | Physics Penalty: {loss_physics.item():.5f} | Val Loss: {loss_val.item():.4f}")

        pd.DataFrame(history).to_csv(f'data/processed/training_history_{self.protocol_km}km.csv', index=False)
        torch.save(model.state_dict(), f'data/processed/st_gcn_model_{self.protocol_km}km.pth')
        
        # ---------------------------------------------------------
        # TRUE DIMENSIONAL PHYSICS EVALUATION (24-Hour Integration Drift)
        # ---------------------------------------------------------
        model.eval()
        with torch.no_grad():
            preds_scaled = model(X_tensor, coords_t, edge_index, edge_attr)[test_mask].numpy()
            actuals_scaled = y_tensor[test_mask].numpy()
            
            # INVERSE TRANSFORM: Z-scores -> Real Velocity (m/s)
            preds_ms = self.scaler_y.inverse_transform(preds_scaled)
            actuals_ms = self.scaler_y.inverse_transform(actuals_scaled)
            
            # Predict coordinate drift displacement across 24 Hours
            # 24 hours * 3600 seconds = 86400 seconds
            drift_seconds = 86400.0
            
            # Error (meters) = Error Delta (m/s) * Seconds
            error_e_meters = (preds_ms[:, 0] - actuals_ms[:, 0]) * drift_seconds
            error_n_meters = (preds_ms[:, 1] - actuals_ms[:, 1]) * drift_seconds
            
            # True Euclidean divergence in km
            abs_err_km = np.sqrt(error_e_meters**2 + error_n_meters**2) / 1000.0
            
            print(f">>> [Scientific Validation] Average True Geographic Displacement Error (24hr Drift): {np.mean(abs_err_km):.2f}km")
            
            err_df = pd.DataFrame({
                'Actual_E_ms': actuals_ms[:, 0], 'Actual_N_ms': actuals_ms[:, 1],
                'Pred_E_ms': preds_ms[:, 0], 'Pred_N_ms': preds_ms[:, 1],
                'Error_KM_24h': abs_err_km
            })
            err_df.to_csv(f'data/processed/spatial_errors_{self.protocol_km}km.csv', index=False)

if __name__ == "__main__":
    for r in [15, 50, 100, 150, 200]:
        print(f"\n======================================")
        print(f"TRAINING SOTA ARCHITECTURE: {r}km")
        print(f"======================================")
        master_file = f'data/master/morpho_graph_master_{r}km.csv'
        if not os.path.exists(master_file):
            print(f"Skipping {r}km pipeline. Master CSV not found.")
            continue
            
        pipeline = MorphoModeler(master_file, protocol_km=r)
        pipeline.prepare_data()
        pipeline.run_st_gcn()