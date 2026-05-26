import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_metrics(actual_e, actual_n, pred_e, pred_n, drift_errors):
    actual_flat = np.concatenate([actual_e, actual_n])
    pred_flat = np.concatenate([pred_e, pred_n])
    
    rmse = np.sqrt(mean_squared_error(actual_flat, pred_flat))
    mae = mean_absolute_error(actual_flat, pred_flat)
    r2 = r2_score(actual_flat, pred_flat)
    drift = np.mean(drift_errors)
    
    return rmse, mae, drift, r2

def main():
    protocols = [15, 50, 100, 150, 200]
    baselines = ["ADE", "LSTM", "CNN-LSTM", "GraphSAGE", "ST-GCN"]
    results = []

    print("Generating final comparison table...")
    
    for protocol in protocols:
        # 1. Main Model (MorphoPINN)
        main_file = f"data/processed/spatial_errors_{protocol}km.csv"
        if os.path.exists(main_file):
            df = pd.read_csv(main_file)
            if len(df) > 0:
                rmse, mae, drift, r2 = compute_metrics(
                    df['Actual_E_ms'], df['Actual_N_ms'],
                    df['Pred_E_ms'], df['Pred_N_ms'],
                    df['Error_KM_24h']
                )
                results.append({
                    "Protocol_km": protocol,
                    "Model": "MorphoPINN",
                    "RMSE_ms": round(rmse, 4),
                    "MAE_ms": round(mae, 4),
                    "Mean_24hr_Drift_Error_km": round(drift, 2),
                    "R2": round(r2, 4),
                    "R2_Warning": "below_mean_baseline" if r2 < 0 else ""
                })

        # 2. Baselines
        for model_name in baselines:
            base_file = f"data/processed/spatial_errors_{model_name}_{protocol}km.csv"
            if os.path.exists(base_file):
                df = pd.read_csv(base_file)
                if len(df) > 0:
                    rmse, mae, drift, r2 = compute_metrics(
                        df['Actual_E'], df['Actual_N'],
                        df['Pred_E'], df['Pred_N'],
                        df['Error_KM']
                    )
                    results.append({
                        "Protocol_km": protocol,
                        "Model": model_name,
                        "RMSE_ms": round(rmse, 4),
                        "MAE_ms": round(mae, 4),
                        "Mean_24hr_Drift_Error_km": round(drift, 2),
                        "R2": round(r2, 4),
                        "R2_Warning": "below_mean_baseline" if r2 < 0 else ""
                    })

    df_res = pd.DataFrame(results)
    
    # Sort for cleaner reading
    df_res = df_res.sort_values(["Protocol_km", "Model"])
    
    os.makedirs('data/processed', exist_ok=True)
    out_path = 'data/processed/final_comparison_table.csv'
    df_res.to_csv(out_path, index=False)
    
    print("\n--- FINAL METRICS COMPARISON TABLE ---")
    print(df_res.to_string(index=False))
    print(f"\n[Success] Table saved to {out_path}")

if __name__ == "__main__":
    main()
