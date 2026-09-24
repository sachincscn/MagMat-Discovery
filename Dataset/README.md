# MagMat Discovery: Dataset Inventory & Data Provenance

This directory contains the curated experimental, literature-mined, and crystallographic databases powering the **MagMat Discovery** machine learning pipeline.

---

## 1. Source Repositories & Data Files

| File Name | Primary Source | Link / Reference | Record Count | Physical Properties |
| :--- | :--- | :--- | :---: | :--- |
| **`Classification_FM_AFM_NM.csv`** | Northeast Materials Database (NEMAD) | [*Nature Communications* (2025)](https://doi.org/10.1038/s41467-025-64458-z) | 35,045 | Magnetic Ground State Phase: Ferromagnetic (22,019), Antiferromagnetic (7,472), Non-Magnetic (5,554) |
| **`FM_with_curie.csv`** | NEMAD | [*Nature Communications* (2025)](https://doi.org/10.1038/s41467-025-64458-z) | 15,518 | Experimental Curie Temperatures ($T_C$ in Kelvin) |
| **`AFM_with_Neel.csv`** | NEMAD | [*Nature Communications* (2025)](https://doi.org/10.1038/s41467-025-64458-z) | 7,734 | Experimental Néel Temperatures ($T_N$ in Kelvin) |
| **`magnetic_anisotropy_materials.csv`** | Peer-Reviewed Literature Compendium | [*J. Magn. Magn. Mater.* & *IEEE Trans. Magn.*](https://doi.org/10.1016/j.jmmm.2017.08.082) | 3,938 | Uniaxial Anisotropy $K_1$ ($\text{J/m}^3$), Coercivity $H_c$ ($\text{A/m}$), Energy Product $(BH)_{\max}$ ($\text{kJ/m}^3$), $H_A$ |
| **`magnetic_materials.csv`** | Literature & Landolt-Börnstein | [*Springer Landolt-Börnstein*](https://materials.springer.com/) | 6,820 | Paramagnetic Curie-Weiss intercept ($\theta_p$ in K), Saturation Magnetization ($M_s$ in $\text{emu/g}$) |
| **`mp_alloys_point_groups.csv`** | Materials Project (v2024.1) | [Materials Project](https://materialsproject.org/) | 132,336 | DFT-relaxed structures: Space Groups (1–230), Point Groups, Wyckoff positions, Density, $E_{\text{hull}}$ |
| **`novomag_refree_magnets.csv`** | NovoMag Database (Ames Lab / Iowa State) | [NovoMag Database](https://www.novomag.physics.iastate.edu/structure-database) | ~500 | Rare-earth-free permanent magnets: $\text{Fe}_{16}\text{N}_2$, $\text{Fe}_3\text{Co}_3X_2$, $\text{MnBi}$, $\text{FePt}$, $\text{Co}_5\text{Zr}$ |

---

## 2. How Data Sources Are Combined & Harmonized

The unified 51,394-compound master training dataset (`output/featurization/features_master.csv`) is constructed in `src/preprocessing.py` through four sequential, leak-free stages:

```
[NEMAD Ground State]  [NEMAD TC / TN]  [Anisotropy & Hc]  [NovoMag RE-Free]
        │                     │                 │                 │
        └──────────────┬──────┴─────────────────┴─────────────────┘
                       ▼
           1. Stoichiometry & Formula Sanitization (pymatgen)
                       ▼
           2. Physics-Informed Unit Standardization (|Hc| > 0, K1 J/m³, TC K)
                       ▼
           3. Ranked Polymorphic Matching against Materials Project DFT (E_hull)
                       ▼
           4. Unified Master Feature Matrix (features_master.csv)
```

### Stage 1: Stoichiometry & Formula Sanitization
- Chemical formulas are parsed and normalized via `pymatgen.core.Composition.reduced_formula`.
- Minerals and exact synonyms are mapped to unambiguous chemical formulas.
- Ambiguous trade family names without determined stoichiometry (e.g. `Nd-Fe-B + 0.1 wt% S`, `Alnico`) are strictly filtered out (`(None, 'failed')`).

### Stage 2: Physics-Informed Unit Standardization
- **Temperatures ($T_C, T_N, \theta_p$)**: Cleaned from ranges and inequalities (`~`, `>`, `±`), converted from Celsius where indicated, and bounded to physical Kelvin ranges ($1\text{ K} \le T \le 1800\text{ K}$).
- **Coercivity ($H_c$)**: Literature records using reverse-field negative notations are converted to physical magnitude ($H_c \equiv |H_c| > 0$). Units ($\text{Oe}, \text{kOe}, \text{T}, \text{kA/m}, \text{MA/m}$) are strictly converted to SI $\text{A/m}$.
- **Anisotropy ($K_1$)**: Converted from $\text{erg/cm}^3, \text{MGOe}, \text{MJ/m}^3$ to SI $\text{J/m}^3$. Transduction $K_1 = \frac{1}{2} \mu_0 H_A M_s$ is applied when explicit $(H_A, M_s)$ pairs are verified.
- **Saturation Magnetization ($M_s$)**: Converted to mass magnetization $\text{emu/g}$ using density $\rho$, and polarization $J = \mu_0 M_s$ (Tesla).
- **Energy Product ($(BH)_{\max}$)**: Converted from $\text{MGOe}$ to $\text{kJ/m}^3$ ($1\text{ MGOe} = 7.9577\text{ kJ/m}^3$).

### Stage 3: Ranked Polymorphic Structure Matching (`merge_point_groups`)
Experimental measurements frequently omit the crystal structure or space group. Each experimental formula is matched to the Materials Project crystal repository using a ranked scoring algorithm:
$$S = 0.50 \cdot \left(1 - \frac{E_{\text{hull}}}{0.10}\right) + 0.30 \cdot \left(1 - \frac{|\rho_{\text{exp}} - \rho_{\text{DFT}}|}{\rho_{\text{exp}}}\right) + 0.20 \cdot \mathbb{I}(\text{SG}_{\text{match}})$$
This matches each experimental measurement to its thermodynamically stable ground-state polymorph ($E_{\text{hull}} \le 0.08\text{ eV/atom}$) without chemical composition leakage.

### Stage 4: Strict Leak-Free Aggregation (No Median Broadcasting)
- Measurements from different laboratories and microstructures for the same compound preserve their genuine experimental variance; **zero median broadcasting** and **zero synthetic imputation** are performed.
- All downstream cross-validation strictly splits groups on `reduced_formula` via 5-fold `GroupKFold`.
