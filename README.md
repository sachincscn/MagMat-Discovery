# MagMat-Discovery

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](app/main.py)

**MagMat-Discovery** is an end-to-end, physics-grounded machine learning framework for magnetic materials discovery and inverse design. Trained on harmonized experimental magnetic records and integrated with crystallographic structures, it evaluates alloy compositions against fundamental condensed-matter boundaries to identify high-performance, rare-earth-free permanent magnets.

---

## Key Capabilities

- **Physics-Grounded Architecture**: Combines a compositional graph neural network (**MagFormer**) with domain physics descriptors (Slater-Pauling band filling, Goodenough-Kanamori superexchange, Stoner criteria, and Kronmüller micromagnetic limits).
- **Multi-Target Prediction**: Jointly models 8 core magnetic targets across phase classification, ordering temperatures ($T_C, T_N, \theta_p$), and extrinsic hysteresis properties ($M_s, H_c, (BH)_{\max}, K_1$).
- **Calibrated Uncertainty**: Delivers rigorous conformal prediction intervals and ensemble standard deviations for reliable discovery.
- **Interactive Web Interfaces**: Streamlit applications offering real-time screening, side-by-side M-H hysteresis tracing, interactive 3D crystal structures, and reality-gap analysis.

---

## Model Performance

Evaluated via 5-fold `GroupKFold` grouped by reduced chemical formula (zero composition leakage across folds):

| Target | Description | Units | Metric | Value |
| :--- | :--- | :--- | :--- | ---: |
| **Phase** | Ground-state ordering (FM / AFM / NM) | — | Accuracy / Macro $F_1$ | **91.9%** / **0.908** |
| **$T_C$** | Curie temperature | K | $R^2$ (MAE) | **0.816** (69.8 K) |
| **$T_N$** | Néel temperature | K | $R^2$ (MAE) | **0.741** (48.6 K) |
| **$M_s$** | Saturation magnetization | emu/g | $R^2$ (MAE) | **0.539** (26.1 emu/g) |
| **$H_c$** | Coercivity | A/m | $R^2$ | **0.622** |
| **$(BH)_{\max}$** | Maximum energy product | kJ/m³ | $R^2$ | **0.507** |
| **$K_1$** | Magnetocrystalline anisotropy magnitude | J/m³ | $R^2$ ($\log_{10}|K_1|$) | **0.461** |
| **$\theta_p$** | Paramagnetic Curie temperature | K | $R^2$ (MAE) | **0.607** (64.7 K) |

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/Sachinscnpdl/MagMat-Discovery.git
cd MagMat-Discovery
pip install numpy pandas scipy scikit-learn lightgbm xgboost catboost pymatgen matminer torch streamlit plotly
```

### 2. Launch Interactive Web Applications

```bash
# Option A: Full Multi-Task Discovery Platform
streamlit run app/main.py

# Option B: Streamlined 2-Tab Concept Studio (Recommended)
streamlit run app/concept_app.py
```

### 3. Command-Line Inference

Predict magnetic phase and all properties for any formula:
```bash
python run_pipeline.py --predict "Nd2Fe14B"
python run_pipeline.py --predict "Fe16N2"
python run_pipeline.py --predict "MnAl"
```

### 4. Running the Pipeline

```bash
# Run complete end-to-end pipeline (stages 1 to 6)
python run_pipeline.py --stage all

# Run specific stages
python run_pipeline.py --stage 4    # Model training & cross-validation
python run_pipeline.py --stage 5    # Inference on Materials Project candidate pool
python run_pipeline.py --stage 6    # Multi-objective permanent magnet screening
```

---

## Repository Structure

```
MagMat-Discovery/
├── app/                         # Interactive Streamlit Web Applications
│   ├── concept_app.py           # Streamlined 2-tab screening & reality gap studio
│   ├── main.py                  # Full multi-task discovery platform
│   ├── data.py                  # High-performance caching & plotting utilities
│   └── theme.py                 # Modern CSS and responsive styling
├── src/                         # Core machine learning & physics library
│   ├── composition_gnn.py       # MagFormer compositional GNN architecture
│   ├── modeling.py              # Multi-model ensembles & conformal calibration
│   ├── cascade.py               # Out-of-fold cross-target physical cascade
│   ├── pinn_hysteresis.py       # Stoner-Wohlfarth & Kronmüller micromagnetics
│   ├── featurization.py         # Physics, Matminer, and structural features
│   ├── preprocessing.py         # Dataset integration & leakage sanitization
│   ├── generative_screening.py  # Inverse composition screening engine
│   └── visualization.py         # Publication figure generators
├── documentation/               # Technical documentation, LaTeX reports, and figures
├── Dataset/                     # Curated experimental and crystal datasets
├── output/                      # Evaluated predictions, caches, and benchmarks
└── run_pipeline.py              # Main orchestrator CLI
```

---

## Citation

```bibtex
@article{poudel2026magmat,
  title={MagMat: Physics-Grounded Screening and Discovery Platform for Permanent Magnets},
  author={Poudel, Sachin},
  year={2026}
}
```