"""
Physics-Informed Hysteresis Operator for Magnetic Pipelines.
Formulates thermodynamic and magnetostatic coupling between:
- Saturation Magnetization (Ms, emu/g and A/m)
- Magnetocrystalline Anisotropy (K1, J/m^3)
- Coercivity (Hc, A/m)
- Energy Product ((BH)max, kJ/m^3)

Based on Stoner-Wohlfarth coherent rotation, Brown's paradox,
and Kronmüller's micromagnetic relation for permanent magnets.
"""

import numpy as np
import pandas as pd

MU_0 = 4.0 * np.pi * 1e-7  # N/A^2 or H/m

# Order-of-magnitude bridge from the Stevens crystal-field torque proxy to an
# absolute anisotropy constant in J/m^3.
#
# stevens_k1_torque is a *relative* single-ion indicator in arbitrary Stevens
# units (median |value| ~1.2e-4). It was being substituted directly for K1 in
# the Stoner-Wohlfarth relations below, which expect J/m^3 (experimental median
# ~5e4). The resulting kappa was ~1e-8, so tanh(kappa/0.5) saturated at zero and
# pinn_loop_squareness sat at its 0.20 floor for every compound -- which in turn
# flattened bh_max_extrinsic, bh_max_bound, log_bh_bound and hardness_parameter.
#
# This is a fixed round constant chosen to land the proxy in the canonical
# 1e2-1e7 J/m^3 anisotropy band. It is NOT fitted to any target: it rescales a
# feature only, and every downstream relation here is monotone in K1. Whenever a
# calibrated K1 is available (use_cascade_preds=True, cascade_pred_k1 in J/m^3)
# that value is used instead and this constant plays no role.
STEVENS_TORQUE_TO_J_M3 = 1.0e8

def compute_pinn_hysteresis_features(df, use_cascade_preds=False):
    """
    Computes rigorous physical demagnetization hysteresis bounds:
    1. pinn_loop_squareness: S = Mr/Ms in [0.20, 0.98]
    2. pinn_bh_max_ideal: Theoretical upper ceiling (1/4 mu_0 (Ms*S)^2) in kJ/m^3
    3. pinn_bh_max_extrinsic: Coercivity operating ceiling (mu_0 Hc (Ms*S) * 0.35) in kJ/m^3
    4. pinn_bh_max_bound: Pinched thermodynamic envelope min(ideal, extrinsic)
    5. pinn_log_bh_bound: log10(1 + pinn_bh_max_bound)
    6. pinn_kronmuller_hk: Anisotropy field 2 K1 / (mu_0 Ms) in A/m
    7. pinn_kronmuller_hc_log: log10(max(10, 0.30 HK - 0.20 Ms)) in A/m
    8. pinn_hardness_parameter: Dimensionless 2 K1 / (mu_0 Ms^2)
    """
    n = len(df)
    
    # 1. Density in g/cm^3 (default 7.5 g/cm^3 for ferromagnetic alloys)
    if 'Material_Density' in df.columns:
        density_s = pd.to_numeric(df['Material_Density'], errors='coerce').fillna(7.5).clip(lower=2.0, upper=20.0)
        density = density_s.values
    else:
        density = np.full(n, 7.5)
        
    # 2. Saturation Magnetization Ms
    if use_cascade_preds and 'cascade_pred_ms' in df.columns:
        ms_s = pd.to_numeric(df['cascade_pred_ms'], errors='coerce').fillna(50.0).clip(lower=0.1, upper=350.0)
    elif 'dft_jarvis_magmom' in df.columns:
        # Fallback estimation from DFT magnetic moment or Magpie GSmagmom
        magmom_s = pd.to_numeric(df['dft_jarvis_magmom'], errors='coerce').fillna(1.0).clip(lower=0.0)
        ms_s = (magmom_s * 5585.0 / 55.85).clip(lower=0.1, upper=350.0)
    elif 'MagpieData maximum GSmagmom' in df.columns:
        magmom_s = pd.to_numeric(df['MagpieData maximum GSmagmom'], errors='coerce').fillna(1.0).clip(lower=0.0)
        ms_s = (magmom_s * 5585.0 / 55.85).clip(lower=0.1, upper=350.0)
    else:
        ms_s = pd.Series(np.full(n, 50.0), index=df.index)
    ms_emu_g = ms_s.values
        
    # Convert Ms from emu/g to A/m and Tesla
    # Ms [A/m] = Ms [emu/g] * density [g/cm^3] * 1000
    # Js [Tesla] = mu_0 * Ms [A/m] = 4*pi*1e-4 * density * Ms [emu/g]
    ms_a_m = ms_emu_g * density * 1000.0
    js_tesla = MU_0 * ms_a_m
    
    # 3. Magnetocrystalline Anisotropy K1
    if use_cascade_preds and 'cascade_pred_k1' in df.columns:
        k1_s = pd.to_numeric(df['cascade_pred_k1'], errors='coerce').fillna(1e4)
    elif 'stevens_k1_torque' in df.columns:
        # Arbitrary Stevens units -> J/m^3; see STEVENS_TORQUE_TO_J_M3.
        k1_s = (pd.to_numeric(df['stevens_k1_torque'], errors='coerce').fillna(0.0)
                * STEVENS_TORQUE_TO_J_M3)
        k1_s = k1_s.where(k1_s.abs() > 0.0, 1e4)
    else:
        k1_s = pd.Series(np.full(n, 1e4), index=df.index)
    k1_j_m3 = k1_s.values
    k1_pos = np.maximum(k1_j_m3, 0.0)
    
    # 4. Anisotropy Field HK and Magnetic Hardness Parameter kappa
    # HK = 2 K1 / (mu_0 Ms)
    hk_a_m = (2.0 * k1_pos) / (js_tesla + 1e-6)
    # kappa = HK / Ms = 2 K1 / (mu_0 Ms^2)
    kappa_hard = (2.0 * k1_pos) / (MU_0 * (ms_a_m ** 2) + 1e-6)
    
    # 5. Stoner-Wohlfarth Loop Squareness S in [0.20, 0.98]
    # S = 0.20 + 0.78 * tanh(kappa / 0.50)
    squareness = 0.20 + 0.78 * np.tanh(kappa_hard / 0.50)
    
    # Remanent polarization Br [Tesla]
    br_tesla = js_tesla * squareness
    
    # 6. Theoretical Remanence-Limited (BH)max Ceiling in kJ/m^3
    # Ideal (BH)max = Br^2 / (4 mu_0) * 1e-3 kJ/m^3 = 198.94367 * Br^2
    bh_ideal_kj_m3 = (br_tesla ** 2) / (4.0 * MU_0 * 1000.0)
    
    # 7. Coercivity Hc
    # In cascading, use predicted Hc if available. Never use experimental ground-truth clean_Hc_A_m!
    if use_cascade_preds and 'cascade_pred_hc' in df.columns:
        hc_s = pd.to_numeric(df['cascade_pred_hc'], errors='coerce').fillna(1e4).clip(lower=10.0, upper=1e8)
    elif use_cascade_preds and 'cascade_pred_hc_log' in df.columns:
        hc_log_s = pd.to_numeric(df['cascade_pred_hc_log'], errors='coerce').fillna(4.0)
        hc_s = (10.0 ** hc_log_s).clip(lower=10.0, upper=1e8)
    else:
        # Analytical Kronmuller estimate from HK and Ms
        hc_s = pd.Series(np.maximum(10.0, 0.30 * hk_a_m - 0.20 * ms_a_m), index=df.index)
    hc_a_m = hc_s.values
        
    # Extrinsic Coercivity-Limited (BH)max Ceiling in kJ/m^3
    # Extrinsic = Br * Hc * f_shape * 1e-3, with f_shape ~ 0.35
    bh_extrinsic_kj_m3 = (br_tesla * hc_a_m * 0.35) / 1000.0
    
    # 8. Pinched Thermodynamic Envelope min(ideal, extrinsic)
    bh_bound_kj_m3 = np.minimum(bh_ideal_kj_m3, bh_extrinsic_kj_m3)
    log_bh_bound = np.log10(1.0 + np.maximum(bh_bound_kj_m3, 0.0))
    
    # 9. Kronmuller Coercivity Features
    # Hc = alpha_K * HK - N_eff * Ms. The nucleation margin is negative whenever
    # stray-field demagnetisation outweighs the anisotropy field, which is the
    # normal situation for soft and moderately hard materials -- and is also the
    # case for every row when HK is built from the Stevens proxy, whose arbitrary
    # units sit ~8 orders of magnitude below J/m^3. Clamping the margin at 10 A/m
    # before taking the log therefore collapsed pinn_kronmuller_hc_log to the
    # constant 1.0 across all 50,159 rows, i.e. the feature carried no signal.
    #
    # The signed margin is retained on a symlog scale so its ordering survives.
    # Tree learners are invariant to the monotone scale error in HK, so no
    # data-derived calibration constant is introduced here; only the information
    # destroyed by the clamp is recovered.
    kron_margin = 0.30 * hk_a_m - 0.20 * ms_a_m
    kron_margin_symlog = np.sign(kron_margin) * np.log10(1.0 + np.abs(kron_margin))
    log_kron_hc = kron_margin_symlog
    log_hk = np.log10(1.0 + hk_a_m)
    
    res = pd.DataFrame({
        'pinn_loop_squareness': squareness,
        'pinn_bh_max_ideal': bh_ideal_kj_m3,
        'pinn_bh_max_extrinsic': bh_extrinsic_kj_m3,
        'pinn_bh_max_bound': bh_bound_kj_m3,
        'pinn_log_bh_bound': log_bh_bound,
        'pinn_kronmuller_hk': log_hk,
        'pinn_kronmuller_hc_log': log_kron_hc,
        'pinn_kronmuller_margin': kron_margin_symlog,
        'pinn_hardness_parameter': np.log10(1.0 + kappa_hard)
    }, index=df.index)
    
    return res
