"""
Ablation study for MorphoPINN at the 50 km protocol.

Six variants, each training with one component removed or modified:
  1. Full MorphoPINN          — all components enabled (baseline)
  2. No divergence penalty    — lambda_div = 0
  3. No Physics-SMOTE         — skip augmentation entirely
  4. No chrono-kinematic enc  — replace Day_Sin / Day_Cos / Lunar_Phase with raw day-of-year integer
  5. K=5 neighbour constraint — under-connected graph
  6. K=20 neighbour constraint — over-smoothed graph

Results saved to data/processed/ablation_results_50km.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree
from sklearn.metrics import mean_squared_error, r2_score

# Bring in shared helpers from model_training without triggering the __main__ block
sys.path.insert(0, os.path.dirname(__file__))
from model_training import MorphoModeler, MorphoSTGNN

EARTH_RADIUS_KM = 6371.0
DRIFT_SECONDS   = 86400.0
PROTOCOL_KM     = 50
MASTER_FILE     = f"data/master/morpho_graph_master_{PROTOCOL_KM}km.csv"
EPOCHS          = 150
BOOTSTRAP_N     = 200   # reduced from 1000 so the script finishes in reasonable time


# ── helpers ──────────────────────────────────────────────────────────────────

def build_graph(coords, y, k):
    """Haversine BallTree graph construction — extracted from MorphoModeler."""
    n = len(coords)
    k_actual = min(k, n - 1)
    if k_actual <= 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 3), dtype=torch.float32)

    coords_rad = np.radians(coords)
    tree = BallTree(coords_rad, metric='haversine')
    dist, ind = tree.query(coords_rad, k=k_actual + 1)

    sources, targets, edge_attrs = [], [], []
    omega = 7.2921e-5
    for i in range(n):
        lat1, lon1 = coords[i]
        ve1, vn1   = y[i]
        v1_norm    = math.sqrt(ve1**2 + vn1**2) + 1e-9
        for j_idx in range(1, k_actual + 1):
            j       = ind[i, j_idx]
            lat2, lon2 = coords[j]
            ve2, vn2   = y[j]
            v2_norm    = math.sqrt(ve2**2 + vn2**2) + 1e-9
            if ((ve1 * ve2) + (vn1 * vn2)) / (v1_norm * v2_norm) < 0:
                continue
            dLon    = math.radians(lon2 - lon1)
            bear    = math.atan2(
                math.sin(dLon) * math.cos(math.radians(lat2)),
                math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
                - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dLon)
            )
            true_dist_km = dist[i, j_idx] * EARTH_RADIUS_KM
            sources.append(j); targets.append(i)
            edge_attrs.append([true_dist_km, bear, 2 * omega * math.sin(math.radians(lat1))])

    return torch.tensor([sources, targets], dtype=torch.long), torch.FloatTensor(edge_attrs)


def train_and_eval(X, y, coords, clusters, feature_cols, cmems_present,
                   use_divergence=True, use_smote=True, k_neighbors=10, epochs=EPOCHS):
    """
    Single training run.  Returns (rmse_ms, r2, mean_ade_km).
    """
    n_samples = len(X)
    n_groups  = len(np.unique(clusters))
    train_mask = np.zeros(n_samples, dtype=bool)

    n_splits  = min(5, n_groups) if n_groups >= 2 else 2
    gkf = GroupKFold(n_splits=n_splits)
    train_idx, _ = next(gkf.split(X, y, groups=clusters))
    train_mask[train_idx] = True
    test_mask = ~train_mask

    # SMOTE augmentation (training fold only)
    if use_smote:
        from model_training import MorphoModeler as _MM
        _dummy = _MM.__new__(_MM)
        aug_X, aug_y, aug_coords = _dummy.apply_physics_smote(
            X[train_mask], y[train_mask], coords[train_mask],
            target_n=max(10, train_mask.sum() // 5)
        )
        n_synth = len(aug_X) - train_mask.sum()
        if n_synth > 0:
            X      = np.vstack([X,      aug_X[-n_synth:]])
            y      = np.vstack([y,      aug_y[-n_synth:]])
            coords = np.vstack([coords, aug_coords[-n_synth:]])
            train_mask = np.concatenate([train_mask, np.ones(n_synth, dtype=bool)])
            test_mask  = np.concatenate([test_mask,  np.zeros(n_synth, dtype=bool)])

    # Scalers — fit on train only
    scaler_X = StandardScaler(); scaler_y = StandardScaler()
    scaler_X.fit(X[train_mask]);  scaler_y.fit(y[train_mask])

    # CMEMS tensor in target-scaled space
    if cmems_present:
        cmems_raw    = X[:, -2:]
        cmems_tensor = torch.FloatTensor(scaler_y.transform(cmems_raw))
    else:
        cmems_tensor = torch.zeros(len(X), 2)

    X_sc = scaler_X.transform(X)
    y_sc = scaler_y.transform(y)

    edge_index, edge_attr = build_graph(coords, y_sc, k=k_neighbors)

    X_t      = torch.FloatTensor(X_sc)
    y_t      = torch.FloatTensor(y_sc)
    coords_t = torch.FloatTensor(coords)

    model     = MorphoSTGNN(X_sc.shape[1], 3, hidden_dim=64, output_dim=2, protocol_km=PROTOCOL_KM)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.HuberLoss(delta=1.0)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        coords_t.requires_grad_(True)

        out        = model(X_t, coords_t, edge_index, edge_attr)
        loss_data  = criterion(out[train_mask], y_t[train_mask])

        if use_divergence:
            res_u = out[train_mask, 0] - cmems_tensor[train_mask, 0]
            res_v = out[train_mask, 1] - cmems_tensor[train_mask, 1]
            grad_u = torch.autograd.grad(res_u, coords_t,
                                         grad_outputs=torch.ones_like(res_u),
                                         create_graph=True, retain_graph=True)[0]
            grad_v = torch.autograd.grad(res_v, coords_t,
                                         grad_outputs=torch.ones_like(res_v),
                                         create_graph=True, retain_graph=True)[0]
            divergence    = grad_u[train_mask, 1] + grad_v[train_mask, 0]
            loss_div      = 0.05 * torch.mean(divergence**2)
        else:
            loss_div = torch.tensor(0.0)

        loss = loss_data + loss_div
        loss.backward()
        optimizer.step()
        scheduler.step()
        coords_t.requires_grad_(False)

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds_sc  = model(X_t, coords_t, edge_index, edge_attr)[test_mask].numpy()
        actual_sc = y_t[test_mask].numpy()

    preds_ms  = scaler_y.inverse_transform(preds_sc)
    actuals_ms = scaler_y.inverse_transform(actual_sc)

    actual_flat = np.concatenate([actuals_ms[:, 0], actuals_ms[:, 1]])
    pred_flat   = np.concatenate([preds_ms[:, 0],   preds_ms[:, 1]])
    rmse = np.sqrt(mean_squared_error(actual_flat, pred_flat))
    r2   = r2_score(actual_flat, pred_flat)

    err_e_m = (preds_ms[:, 0] - actuals_ms[:, 0]) * DRIFT_SECONDS
    err_n_m = (preds_ms[:, 1] - actuals_ms[:, 1]) * DRIFT_SECONDS
    ade_km  = float(np.mean(np.sqrt(err_e_m**2 + err_n_m**2) / 1000.0))

    return float(rmse), float(r2), ade_km


def bootstrap_eval(X, y, coords, clusters, feature_cols, cmems_present, config, n=BOOTSTRAP_N):
    """Run n bootstrap iterations and return mean metrics."""
    rmses, r2s, ades = [], [], []
    for i in range(n):
        try:
            rmse, r2, ade = train_and_eval(
                X.copy(), y.copy(), coords.copy(), clusters.copy(),
                feature_cols, cmems_present, **config
            )
            rmses.append(rmse); r2s.append(r2); ades.append(ade)
        except Exception as e:
            print(f"  [Bootstrap {i}] failed: {e}")
    return np.mean(rmses), np.mean(r2s), np.mean(ades)


# ── main ─────────────────────────────────────────────────────────────────────

def run_ablation():
    if not os.path.exists(MASTER_FILE):
        print(f"[Error] Master file not found: {MASTER_FILE}. Run etl_pipeline.py first.")
        return

    print(f"\n{'='*60}")
    print(f"ABLATION STUDY — {PROTOCOL_KM}km protocol ({BOOTSTRAP_N}-bootstrap mean)")
    print(f"{'='*60}")

    pipeline = MorphoModeler(MASTER_FILE, protocol_km=PROTOCOL_KM)
    pipeline.prepare_data()

    X       = pipeline.X.copy()
    y       = pipeline.y_reg.copy()
    coords  = pipeline.coords.copy()
    clusters = pipeline.clusters.copy()
    feature_cols   = pipeline.feature_cols
    cmems_present  = 'CMEMS_U' in feature_cols

    # Chrono-kinematic indices for the no-encoding variant
    chrono_cols = ['Day_Sin', 'Day_Cos', 'Lunar_Phase']
    chrono_idx  = [feature_cols.index(c) for c in chrono_cols if c in feature_cols]

    ablation_configs = [
        {
            "label": "Full MorphoPINN (All Components)",
            "X": X, "use_divergence": True, "use_smote": True, "k_neighbors": 10
        },
        {
            "label": "w/o Divergence Continuity Penalty (λ_div=0)",
            "X": X, "use_divergence": False, "use_smote": True, "k_neighbors": 10
        },
        {
            "label": "w/o Physics-SMOTE Augmentation",
            "X": X, "use_divergence": True, "use_smote": False, "k_neighbors": 10
        },
        {
            "label": "w/o Chrono-Kinematic Fourier Encoding",
            "X": None,  # computed below
            "use_divergence": True, "use_smote": True, "k_neighbors": 10
        },
        {
            "label": "K=5 Neighbour Constraint (Under-connected)",
            "X": X, "use_divergence": True, "use_smote": True, "k_neighbors": 5
        },
        {
            "label": "K=20 Neighbour Constraint (Over-smoothed)",
            "X": X, "use_divergence": True, "use_smote": True, "k_neighbors": 20
        },
    ]

    # Build linear-day-of-year feature matrix for variant 4
    X_linear = X.copy()
    if chrono_idx:
        pipeline.prepare_data()  # reload to get original df
        df_raw = pd.read_csv(MASTER_FILE)
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        doy = df_raw['Date'].dt.dayofyear.values.astype(float)
        # Replace all three chrono columns with a single linear DOY in first slot;
        # zero out the remaining slots so feature dim stays the same.
        X_linear[:, chrono_idx[0]] = doy / 365.25   # normalised 0-1
        for idx in chrono_idx[1:]:
            X_linear[:, idx] = 0.0
    ablation_configs[3]["X"] = X_linear

    results = []
    for cfg in ablation_configs:
        label = cfg["label"]
        X_use = cfg["X"]
        print(f"\n[Running] {label} ...")
        rmse, r2, ade = bootstrap_eval(
            X_use, y, coords, clusters, feature_cols, cmems_present,
            config={
                "use_divergence": cfg["use_divergence"],
                "use_smote":      cfg["use_smote"],
                "k_neighbors":    cfg["k_neighbors"],
            }
        )
        print(f"  RMSE={rmse:.3f} m/s  |  R²={r2:.3f}  |  ADE={ade:.1f} km")
        results.append({"Configuration": label, "RMSE_ms": round(rmse, 3), "R2": round(r2, 3), "ADE_km": round(ade, 1)})

    df_out = pd.DataFrame(results)
    os.makedirs('data/processed', exist_ok=True)
    out_path = 'data/processed/ablation_results_50km.csv'
    df_out.to_csv(out_path, index=False)
    print(f"\n{'='*60}")
    print(df_out.to_string(index=False))
    print(f"\n[Success] Ablation results saved to {out_path}")


if __name__ == "__main__":
    run_ablation()
