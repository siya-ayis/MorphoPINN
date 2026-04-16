# 🌊 MorphoPINN: Spatio-Temporal Microplastic Drift Prediction
**A Physics-Informed Spatio-Temporal Graph Attention Network (EPD-PINN) for Global Marine Microplastic Forensic Routing.**

This repository contains the official PyTorch implementation and PyDeck simulation dashboard for the framework detailed in our upcoming journal submission.

## 📌 Project Overview
MorphoPINN completely replaces classical deterministic tracking algorithms with an **Encoder-Processor-Decoder (EPD) Graph Transformer**. By unifying human observational data with mesoscale physical oceanography, the system predicts extreme real-time particle drift governed by strict mass-conservation penalties.

### 🏆 Core Architectural Features
* **Complex Spatial Topology:** Transforms millions of sparse pollution anchors into structured coordinate graphs using a **BallTree indexed K=10 Haversine Adjacency Matrix**, natively respecting Earth-Centered Earth-Fixed (ECEF) curvature.
* **Physics-Informed Structural Bounds:** Supresses unphysical recurrent trajectory hallucinations (e.g., standard LSTM/GraphSAGE unbounded drift) via a **Navier-Stokes `autograd` divergence penalty**.
* **Stochastic Eulerian-Lagrangian Diffusion:** Deploys continuous Fokker-Planck Brownian noise injection mapping to simulate non-deterministic sub-grid dispersion.
* **Apriori Meta-Mining:** Validated via **Fisher's Exact Test ($p < 0.05$)**, discovering a mathematically rigid structural Lift Score of 81.00 directly connecting physical Manta Net interventions with Open Ocean accumulation zones.

## 🗄️ Hydrodynamic Data Integrations
This framework ingests three massive, independent oceanographic arrays:
1. **NCEI Global Marine Microplastics Dataset:** The categorical static spatial anchor matrix.
2. **NOAA GDP Drifter Telemetry:** 21MM+ raw Lagrangian velocity tuples acting as the kinematic ground-truth filtered via a strict > 3.0 m/s scalar rejection audit.
3. **Copernicus (CMEMS) Reanalysis:** Mesoscale Eulerian continuous velocity matrices ensuring flow continuity.

## 🚀 Execution Pipeline

### 1. Environment Deployment
```bash
# Python 3.9+ highly recommended
python -m pip install -r requirements.txt
# Critical Dependencies
python -m pip install torch torchvision torchaudio streamlit pydeck folium networkx scikit-learn
```

### 2. Deep Learning Compute Modules
*Ensure all legacy models are cleared before bootstrapping the Spatio-Temporal protocols (15km-200km).*

**ETL Topological Construction:**
```bash
python src/etl_pipeline.py
```

**EPD-PINN Neural Backpropagation Tracking:**
```bash
python src/model_training.py
```

**Forensic Apriori Extraction:**
```bash
python src/run_apriori.py
```

**Metric Validation (NSE, RMSE, MAPE):**
```bash
python src/calculate_metrics.py
```

### 3. Activating the MorphoPINN Dashboard
Boot up the Streamlit frontend to run real-time PyDeck trajectory forecasting on your local host GPU/CPU hardware.
```bash
python -m streamlit run src/dashboard.py
```
*Access the local interface on **http://localhost:8501***

## 📖 Citation
If you utilize MorphoPINN or the optimized PyDeck visualization suite in your research, please cite our paper:

```bibtex
@article{morphopinn2026,
  title={MorphoPINN: A Physics-Informed Spatio-Temporal Graph Attention Network for Microplastic Drift Prediction in the North Atlantic Basin},
  author={Siya Srivastava, Nehal Rai, and Harshini Rebala},
  journal={Under Review},
  year={2026}
}
```

---
*Built for state-of-the-art computational fluid dynamics and unstructured global pollution tracking.*