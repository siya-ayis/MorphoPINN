import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DRIFT_SECONDS = 86400.0  # 24 hours in seconds — used for ADE calculation across all models


def recompute_ade(df):
    """
    Compute 24-hour geographic displacement error (km) from velocity-error columns.
    Handles both MorphoPINN column names (*_ms suffix) and legacy baseline names.
    Using v_error (m/s) × 86400 s / 1000 m/km = v_error × 86.4.
    """
    if 'Actual_E_ms' in df.columns:
        ae, an = df['Actual_E_ms'].values, df['Actual_N_ms'].values
        pe, pn = df['Pred_E_ms'].values,   df['Pred_N_ms'].values
    else:
        ae, an = df['Actual_E'].values, df['Actual_N'].values
        pe, pn = df['Pred_E'].values,   df['Pred_N'].values

    err_e_m = (pe - ae) * DRIFT_SECONDS
    err_n_m = (pn - an) * DRIFT_SECONDS
    return np.sqrt(err_e_m**2 + err_n_m**2) / 1000.0


def compute_metrics(df):
    if 'Actual_E_ms' in df.columns:
        ae, an = df['Actual_E_ms'].values, df['Actual_N_ms'].values
        pe, pn = df['Pred_E_ms'].values,   df['Pred_N_ms'].values
    else:
        ae, an = df['Actual_E'].values, df['Actual_N'].values
        pe, pn = df['Pred_E'].values,   df['Pred_N'].values

    actual_flat = np.concatenate([ae, an])
    pred_flat   = np.concatenate([pe, pn])

    rmse  = np.sqrt(mean_squared_error(actual_flat, pred_flat))
    mae   = mean_absolute_error(actual_flat, pred_flat)
    r2    = r2_score(actual_flat, pred_flat)
    drift = np.mean(recompute_ade(df))

    return rmse, mae, drift, r2


def main():
    protocols = [15, 50, 100, 150, 200]
    baselines = ["ADE", "LSTM", "CNN-LSTM", "GraphSAGE", "ST-GCN"]
    results = []

    print("Generating final comparison table (unified ADE formula: v_error × 86400 / 1000)...")

    for protocol in protocols:
        # MorphoPINN
        main_file = f"data/processed/spatial_errors_{protocol}km.csv"
        if os.path.exists(main_file):
            df = pd.read_csv(main_file)
            if len(df) > 0:
                rmse, mae, drift, r2 = compute_metrics(df)
                results.append({
                    "Protocol_km": protocol,
                    "Model": "MorphoPINN",
                    "RMSE_ms": round(rmse, 4),
                    "MAE_ms":  round(mae,  4),
                    "Mean_24hr_Drift_Error_km": round(drift, 2),
                    "R2": round(r2, 4),
                    "R2_Warning": "below_mean_baseline" if r2 < 0 else ""
                })

        # Baselines
        for model_name in baselines:
            base_file = f"data/processed/spatial_errors_{model_name}_{protocol}km.csv"
            if os.path.exists(base_file):
                df = pd.read_csv(base_file)
                if len(df) > 0:
                    rmse, mae, drift, r2 = compute_metrics(df)
                    results.append({
                        "Protocol_km": protocol,
                        "Model": model_name,
                        "RMSE_ms": round(rmse, 4),
                        "MAE_ms":  round(mae,  4),
                        "Mean_24hr_Drift_Error_km": round(drift, 2),
                        "R2": round(r2, 4),
                        "R2_Warning": "below_mean_baseline" if r2 < 0 else ""
                    })

    df_res = pd.DataFrame(results).sort_values(["Protocol_km", "Model"])
    os.makedirs('data/processed', exist_ok=True)
    out_path = 'data/processed/final_comparison_table.csv'
    df_res.to_csv(out_path, index=False)

    print("\n--- FINAL METRICS COMPARISON TABLE (corrected ADE) ---")
    print(df_res.to_string(index=False))
    print(f"\n[Success] Table saved to {out_path}")


if __name__ == "__main__":
    main()
