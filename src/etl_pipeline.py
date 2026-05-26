import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import os
import xarray as xr

EARTH_RADIUS_KM = 6371.0 

class MorphoGraphPipeline:
    def __init__(self, plastic_path, drifter_path):
        self.plastic_path = plastic_path
        self.drifter_path = drifter_path
        self.plastics = None
        self.drifters = None
        self.cmems = None
        self.cmems_ds = None
        self.master_table = None

    def load_and_clean_data(self):
        print("[Status] Pillar 1: Loading datasets...")
        
        try:
            self.plastics = pd.read_csv(self.plastic_path)
            col_map = {
                'Sample Date': 'Date',
                'Latitude (degree)': 'Latitude', 
                'Longitude (degree)': 'Longitude',
                'Microplastics Measurement': 'Count',
                'Standardized Nurdle  Amount': 'Nurdle_Count'
            }
            self.plastics.rename(columns=col_map, inplace=True)
            self.plastics['Date'] = pd.to_datetime(self.plastics['Date'], format='mixed', errors='coerce', utc=True).dt.tz_localize(None)
            
            total_initial = len(self.plastics)
            
            # Metric 1.1: Spatio-Temporal Pruning
            self.plastics.dropna(subset=['Latitude', 'Longitude', 'Date'], inplace=True)
            
            # Metric 1.2: Logic Target (Density Drift)
            mask = self.plastics['Count'].isna() & self.plastics['Nurdle_Count'].isna()
            self.plastics = self.plastics[~mask].copy()
            self.plastics['Normalized_Density_Score'] = self.plastics['Count'].fillna(self.plastics['Nurdle_Count'])
            
            # Metric 1.4: Metadata & Ocean Handling
            self.plastics['Ocean'] = self.plastics['Ocean'].fillna('Unknown_Basin')
            
            core_cols = ['Date', 'Latitude', 'Longitude', 'Normalized_Density_Score', 
                         'Sampling Method', 'Ocean', 'Marine Setting', 'Concentration Class']
            
            for c in core_cols:
                if c not in self.plastics.columns: self.plastics[c] = 'Unknown'
            self.plastics = self.plastics[core_cols]
            
            # Save a briefing file to answer the PI immediately
            with open('data/processed/PI_Briefing_Count.txt', 'w') as f:
                f.write(f"Survived Rows: {len(self.plastics)}")
                
            print(f"[Pillar 1 Briefing] True NCEI rows surviving Spatio-Temporal & Target Pruning: {len(self.plastics)} out of {total_initial}")

        except Exception as e:
            print(f"[Critical Error] Failed to load Plastics CSV: {e}")
            return

        print("[Status] Loading Drifter Data (High Memory Usage)...")
        try:
            self.drifters = pd.read_csv(self.drifter_path, 
                                        usecols=['time', 'latitude', 'longitude', 've', 'vn'],
                                        dtype={'latitude': 'float32', 'longitude': 'float32', 've': 'float32', 'vn': 'float32'},
                                        low_memory=False, skiprows=[1])
            self.drifters.rename(columns={'time': 'Timestamp', 'latitude': 'Latitude', 'longitude': 'Longitude'}, inplace=True)
            
            self.drifters.dropna(subset=['Latitude', 'Longitude', 've', 'vn'], inplace=True)
            self.drifters['Timestamp'] = pd.to_datetime(self.drifters['Timestamp'], format='ISO8601', errors='coerce', utc=True).dt.tz_localize(None)
            self.drifters.dropna(subset=['Timestamp'], inplace=True)
        except Exception as e:
            print(f"[Critical Error] Drifter Load Failed: {e}")
            self.drifters = None

        self.load_cmems_velocity()

    def load_cmems_velocity(self):
        print("[Status] Pillar 1.5: Attempting to load CMEMS Eulerian Reanalysis...")
        file_path = 'data/raw/cmems_velocity.nc'
        if not os.path.exists(file_path):
            print(f"[Warning] CMEMS file '{file_path}' not found. Skipping Eulerian integration.")
            return

        try:
            # Open lazily — no data loaded into RAM until .sel()/.load() is called
            ds = xr.open_dataset(file_path)
            print("[CMEMS Dataset Summary]")
            print(ds)

            u_col = 'uo' if 'uo' in ds else 'u' if 'u' in ds else None
            v_col = 'vo' if 'vo' in ds else 'v' if 'v' in ds else None

            if u_col and v_col:
                rename_map = {}
                if u_col != 'uo': rename_map[u_col] = 'uo'
                if v_col != 'vo': rename_map[v_col] = 'vo'
                if rename_map:
                    ds = ds.rename(rename_map)
                self.cmems_ds = ds
                print(f"[Success] CMEMS dataset opened lazily: {dict(ds.dims)}")
            else:
                print("[Warning] Could not identify U/V velocity variables in CMEMS file.")

        except Exception as e:
            print(f"[Warning] Failed to load CMEMS file: {e}")

    def preprocess_drifters(self):
        if self.drifters is None: return
        print("[Status] Pillar 0: Anomaly Purge...")

        self.drifters['velocity_norm'] = np.sqrt(self.drifters['ve']**2 + self.drifters['vn']**2)
        outliers = self.drifters[self.drifters['velocity_norm'] > 3.0]
        self.drifters = self.drifters[self.drifters['velocity_norm'] <= 3.0].copy()
        print(f"[Pillar 0 Enforcement] Dropped {len(outliers)} impossible Lagrangian drift vectors (>3.0m/s).")

        iso = IsolationForest(contamination=0.01, random_state=42) 
        self.drifters['anomaly'] = iso.fit_predict(self.drifters[['ve', 'vn']])
        self.drifters = self.drifters[self.drifters['anomaly'] == 1].drop(columns=['anomaly'])
        
        scaler = StandardScaler()
        self.drifters[['ve_norm', 'vn_norm']] = scaler.fit_transform(self.drifters[['ve', 'vn']])

    def spatiotemporal_join(self, target_radius_km=15.0):
        if self.drifters is None or self.plastics is None: return
        print(f"[Status] Joining for protocol {target_radius_km}km...")
        
        drifter_rad = np.radians(self.drifters[['Latitude', 'Longitude']].values)
        plastic_rad = np.radians(self.plastics[['Latitude', 'Longitude']].values)
        tree = BallTree(drifter_rad, metric='haversine')
        
        # True spherical conversion for Haversine without 111.32 scalar assumption
        search_rad = target_radius_km / EARTH_RADIUS_KM
        indices_list, distances_list = tree.query_radius(plastic_rad, r=search_rad, return_distance=True)

        joined_data = []
        for i, (plastic_idx, neighbor_indices) in enumerate(zip(self.plastics.index, indices_list)):
            if len(neighbor_indices) == 0: continue 
                
            plastic_row = self.plastics.loc[plastic_idx]
            p_time = plastic_row['Date']
            
            potential_matches = self.drifters.iloc[neighbor_indices].copy()
            potential_matches['time_diff_hrs'] = (potential_matches['Timestamp'] - p_time).abs().dt.total_seconds() / 3600.0
            
            valid = potential_matches[potential_matches['time_diff_hrs'] <= 24.0]
            if not valid.empty:
                match = valid.loc[valid['time_diff_hrs'].idxmin()]
                joined_data.append({
                    'Sample_ID': plastic_idx,
                    'Date': p_time,
                    'Latitude': plastic_row['Latitude'],
                    'Longitude': plastic_row['Longitude'],
                    'Density_Score': plastic_row['Normalized_Density_Score'],
                    'Sampling_Method': plastic_row['Sampling Method'],
                    'Ocean': plastic_row['Ocean'],
                    'Marine_Setting': plastic_row['Marine Setting'],
                    'Concentration_Class': plastic_row['Concentration Class'],
                    'Velocity_E': match['ve'],
                    'Velocity_N': match['vn'],
                    'Velocity_E_Norm': match.get('ve_norm', 0),
                    'Velocity_N_Norm': match.get('vn_norm', 0)
                })

        df = pd.DataFrame(joined_data)

        # Chrono-Kinematics
        if not df.empty and 'Date' in df.columns:
            date_series = pd.to_datetime(df['Date'])
            day_of_year = date_series.dt.dayofyear
            df['Day_Sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
            df['Day_Cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
            
            epoch = pd.Timestamp("2000-01-21")
            if date_series.dt.tz is not None: epoch = epoch.tz_localize('UTC')
            df['Lunar_Phase'] = np.sin(((date_series - epoch).dt.total_seconds() / 86400 % 29.53059) / 29.53059 * 2 * np.pi)

        # Pillar 3: Authentic One-Hot Encoding
        if not df.empty:
            for feat in ['Sampling_Method', 'Marine_Setting']:
                dummies = pd.get_dummies(df[feat], prefix=feat)
                for col in dummies.columns:
                    df[col] = dummies[col].astype(float)
        
        self.master_table = df

    def join_cmems_to_master(self):
        if self.master_table is None or self.master_table.empty: return
        if self.cmems_ds is None:
            print("[Warning] No CMEMS data loaded. Filling Eulerian fields with NaN.")
            self.master_table['CMEMS_U'] = np.nan
            self.master_table['CMEMS_V'] = np.nan
            return

        print("[Status] Joining CMEMS Eulerian Velocity fields to Master Table (vectorized xarray)...")

        obs_times = pd.to_datetime(self.master_table['Date'].values)
        obs_lats = self.master_table['Latitude'].values
        obs_lons = self.master_table['Longitude'].values

        # Vectorized pointwise nearest-neighbor selection — loads only the
        # ~22k requested grid cells instead of the full 20 GB dataset
        times_da = xr.DataArray(obs_times, dims='obs')
        lats_da = xr.DataArray(obs_lats, dims='obs')
        lons_da = xr.DataArray(obs_lons, dims='obs')

        try:
            pts = self.cmems_ds.sel(
                time=times_da, latitude=lats_da, longitude=lons_da,
                method='nearest'
            )
            pts = pts.squeeze(drop=True).load()
            self.master_table['CMEMS_U'] = pts['uo'].values.flatten()
            self.master_table['CMEMS_V'] = pts['vo'].values.flatten()
            matched = self.master_table['CMEMS_U'].notna().sum()
            print(f"[Pillar 1.5 Enforcement] CMEMS matched {matched} out of {len(self.master_table)} instances.")
        except Exception as e:
            print(f"[Warning] CMEMS join failed: {e}. Filling with NaN.")
            self.master_table['CMEMS_U'] = np.nan
            self.master_table['CMEMS_V'] = np.nan

    def run_clustering_lab7(self, k=1000):
        # [Constraint 1: Probability Density Weighting (Literature Integration)]
        # Metric 2.1: Eulerian GMMs weighted topologically by legitimate Density Scores
        if self.master_table is None or self.master_table.empty: return
        n_samples = len(self.master_table)
        
        # Scaled up node counts massively to prevent data starvation
        k_actual = min(k, max(1, n_samples // 2))
        if k_actual == 0: return

        coords = self.master_table[['Latitude', 'Longitude']]
        weights = self.master_table['Density_Score'].fillna(1.0).replace([np.inf, -np.inf], 1.0)
        
        # Scikit-learn GMM lacks 'sample_weight'. We resolve this strictly via Probability Density Resampling.
        # The dense patches of microplastic mass must exert 'mathematical gravity' on the ellipsoids.
        weights_norm = (weights / weights.sum()).values
        sampled_indices = np.random.choice(len(coords), size=max(n_samples * 5, 100000), p=weights_norm, replace=True)
        sampled_coords = coords.iloc[sampled_indices]
        
        from sklearn.mixture import GaussianMixture
        self.gmm_model = GaussianMixture(n_components=k_actual, covariance_type='full', random_state=42)
        self.gmm_model.fit(sampled_coords)
        self.master_table['Node_Cluster_ID'] = self.gmm_model.predict(coords)
        
        centroids = pd.DataFrame(self.gmm_model.means_, columns=['Lat', 'Lon'])
        centroids.to_csv('data/processed/graph_nodes_centroids.csv', index_label='Node_ID')

        # Metric 1.3: Topological Collapse 
        # Resolves the 'Overlapping Duplicates' data leak where expanding the radius inflated 
        # NCEI rows matching thousands of drifter pings. Collapses all ping overlaps into strict GMM nodes.
        agg_rules = {
            'Date': 'first',
            'Latitude': 'mean',
            'Longitude': 'mean',
            'Density_Score': 'mean',
            'Sampling_Method': 'first',
            'Ocean': 'first',
            'Marine_Setting': 'first',
            'Concentration_Class': 'first',
            'Velocity_E': 'mean',
            'Velocity_N': 'mean',
            'Velocity_E_Norm': 'mean',
            'Velocity_N_Norm': 'mean',
            'Day_Sin': 'mean',
            'Day_Cos': 'mean',
            'Lunar_Phase': 'mean'
        }
        
        for c in self.master_table.columns:
            if c.startswith('Sampling_Method_') or c.startswith('Marine_Setting_'):
                agg_rules[c] = 'max'
                
        # [Topological Upgrade] Store the collapsed table separately to preserve it for Lab 7 Apriori logic, 
        # but keep the master_table full and un-collapsed so the Deep Neural Net learns from all raw physical points.
        self.collapsed_table = self.master_table.groupby('Node_Cluster_ID').agg(agg_rules).reset_index()

    def prepare_apriori_lab6(self, protocol_km=15):
        if not hasattr(self, 'collapsed_table') or self.collapsed_table is None or self.collapsed_table.empty: return
        
        tx = self.collapsed_table[['Sampling_Method', 'Marine_Setting', 'Concentration_Class', 'Node_Cluster_ID']].copy()
        tx['Sampling_Method'] = 'Method_' + tx['Sampling_Method'].astype(str)
        tx['Marine_Setting'] = 'Setting_' + tx['Marine_Setting'].astype(str)
        tx['Concentration_Class'] = 'Class_' + tx['Concentration_Class'].astype(str)
        tx['Node_Cluster_ID'] = 'Region_' + tx['Node_Cluster_ID'].astype(str)
        
        save_path = f'data/processed/apriori_transactions_{protocol_km}km.csv'
        tx.to_csv(save_path, index=False, header=False)

    def export_master_data(self, suffix=""):
        if self.master_table is not None:
            self.master_table.to_csv(f'data/master/morpho_graph_master{suffix}.csv', index=False)

if __name__ == "__main__":
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/master', exist_ok=True)

    pipeline = MorphoGraphPipeline('data/raw/ncei_microplastics.csv', 'data/raw/gdp_drifter_hourly.csv')
    pipeline.load_and_clean_data()
    pipeline.preprocess_drifters()
    
    # Executing conclusively on the full 5-layer spatial matrix
    for protocol_km in [15, 50, 100, 150, 200]: 
        print(f"\n[Executing {protocol_km}km Protocol]")
        pipeline.spatiotemporal_join(target_radius_km=protocol_km)
        pipeline.join_cmems_to_master()
        pipeline.run_clustering_lab7(k=1000)
        pipeline.prepare_apriori_lab6(protocol_km=protocol_km)
        pipeline.export_master_data(suffix=f"_{protocol_km}km")