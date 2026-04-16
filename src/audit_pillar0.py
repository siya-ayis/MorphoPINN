import pandas as pd
import numpy as np
import time

def run_audit():
    t0 = time.time()
    print("--- PILLAR 0: DATA AUDIT START ---")
    print("[Status] Loading NCEI Microplastics Data...")
    try:
        ncei = pd.read_csv('data/raw/ncei_microplastics.csv')
        ncei.rename(columns={'Sample Date': 'Date', 'Latitude (degree)': 'Latitude', 'Longitude (degree)': 'Longitude'}, inplace=True)
        ncei['Date'] = pd.to_datetime(ncei['Date'], format='mixed', errors='coerce', utc=True).dt.tz_localize(None)
        
        ncei_min_time, ncei_max_time = ncei['Date'].min(), ncei['Date'].max()
        ncei_min_lat, ncei_max_lat = ncei['Latitude'].min(), ncei['Latitude'].max()
        ncei_min_lon, ncei_max_lon = ncei['Longitude'].min(), ncei['Longitude'].max()
        
        print(f"[NCEI] Time Bounds: {ncei_min_time} to {ncei_max_time}")
        print(f"[NCEI] Spatial Bounds: Lat({ncei_min_lat:.2f}, {ncei_max_lat:.2f}), Lon({ncei_min_lon:.2f}, {ncei_max_lon:.2f})")
    except Exception as e:
        print(f"[Critical Error] Failed to load NCEI data: {e}")
        return

    print("\n[Status] Loading GDP Drifter Hourly Data (Fast Vectorized Parsing)...")
    try:
        # Load exactly what we need, skipping the 2nd row (index 1) which contains string units
        drifters = pd.read_csv('data/raw/gdp_drifter_hourly.csv', 
                               usecols=['time', 'latitude', 'longitude', 've', 'vn'],
                               dtype={'latitude': 'float32', 'longitude': 'float32', 've': 'float32', 'vn': 'float32'},
                               low_memory=False,
                               skiprows=[1])
        drifters.rename(columns={'time': 'Timestamp', 'latitude': 'Latitude', 'longitude': 'Longitude'}, inplace=True)
        
        drifters.dropna(subset=['Latitude', 'Longitude', 've', 'vn', 'Timestamp'], inplace=True)
        
        # Fast datetime parse
        drifters['Timestamp'] = pd.to_datetime(drifters['Timestamp'], format='ISO8601', errors='coerce', utc=True).dt.tz_localize(None)
        drifters.dropna(subset=['Timestamp'], inplace=True)
        
        drift_min_time, drift_max_time = drifters['Timestamp'].min(), drifters['Timestamp'].max()
        drift_min_lat, drift_max_lat = drifters['Latitude'].min(), drifters['Latitude'].max()
        drift_min_lon, drift_max_lon = drifters['Longitude'].min(), drifters['Longitude'].max()
        
        print(f"[GDP] Time Bounds: {drift_min_time} to {drift_max_time}")
        print(f"[GDP] Spatial Bounds: Lat({drift_min_lat:.2f}, {drift_max_lat:.2f}), Lon({drift_min_lon:.2f}, {drift_max_lon:.2f})")
    except Exception as e:
        print(f"[Critical Error] Failed to load Drifter data: {e}")
        return

    print("\n--- METRIC 0.1: OUTLIERS ---")
    drifters['velocity_norm'] = np.sqrt(drifters['ve']**2 + drifters['vn']**2)
    outliers = drifters[drifters['velocity_norm'] > 3.0]
    outlier_count = len(outliers)
    print(f"Quarantined rows (> 3.0 m/s): {outlier_count} rows")
    if outlier_count > 0:
        print(f"Maximum velocity detected: {outliers['velocity_norm'].max():.2f} m/s")

    print("\n--- METRIC 0.2: COVERAGE OVERLAP ---")
    t_start = max(drift_min_time, ncei_min_time)
    t_end = min(drift_max_time, ncei_max_time)
    overlap_time = t_end - t_start
    
    overlap_lat = max(0, min(drift_max_lat, ncei_max_lat) - max(drift_min_lat, ncei_min_lat))
    overlap_lon = max(0, min(drift_max_lon, ncei_max_lon) - max(drift_min_lon, ncei_min_lon))
    
    print(f"Temporal Overlap Duration: {overlap_time}")
    print(f"Latitudinal Overlap Distance: {overlap_lat:.2f} degrees")
    print(f"Longitudinal Overlap Distance: {overlap_lon:.2f} degrees")
    
    if overlap_time.total_seconds() > 0 and overlap_lat > 0 and overlap_lon > 0:
        print("OVERLAP VERIFIED: TRUE (Both Datasets Intersect Spatially and Temporally)")
    else:
        print("OVERLAP VERIFIED: FALSE (Zero intersection detected. Pipeline Halt Condition Met.)")

    print(f"\nAudit completed in {time.time() - t0:.1f} seconds")

if __name__ == "__main__":
    run_audit()
