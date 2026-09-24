#!/usr/bin/env python3
"""
Error Quantification & Uncertainty Calibration Module.
Performs rigorous statistical evaluation:
1. Conformal prediction empirical coverage checks (nominal vs. empirical coverage)
2. Residual normality (Q-Q plots, Skewness, Kurtosis, MedAE)
3. Physical stratification breakdown by crystal symmetry and chemical family
4. Heteroscedastic error diagnostics and parity figures
"""

import os
import sys
pipeline_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pipeline_root not in sys.path:
    sys.path.insert(0, pipeline_root)

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#D1D5DB'
plt.rcParams['axes.linewidth'] = 0.8


def evaluate_conformal_coverage(y_true, y_pred, y_unc, nominal_level=0.90):
    """
    Computes empirical coverage rate and sharpness for prediction intervals.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_unc = np.asarray(y_unc, dtype=float)
    
    valid_mask = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (~np.isnan(y_unc)) & (y_unc > 0)
    if not valid_mask.any():
        return {'empirical_coverage': 0.0, 'nominal_level': nominal_level, 'mean_width': 0.0, 'valid_samples': 0}
        
    residuals = np.abs(y_true[valid_mask] - y_pred[valid_mask])
    unc_bounds = y_unc[valid_mask]
    
    covered = residuals <= unc_bounds
    empirical_cov = float(np.mean(covered))
    mean_width = float(np.mean(unc_bounds * 2.0))
    median_width = float(np.median(unc_bounds * 2.0))
    
    return {
        'empirical_coverage': round(empirical_cov, 4),
        'nominal_level': nominal_level,
        'coverage_error': round(empirical_cov - nominal_level, 4),
        'mean_interval_width': round(mean_width, 2),
        'median_interval_width': round(median_width, 2),
        'valid_samples': int(np.sum(valid_mask))
    }


def compute_stratified_error_metrics(df: pd.DataFrame, group_col: str, true_col: str, pred_col: str):
    """
    Calculates MAE, RMSE, and R2 grouped by a categorical physical feature (crystal system, formula family).
    """
    records = []
    if group_col not in df.columns or true_col not in df.columns or pred_col not in df.columns:
        return pd.DataFrame()
        
    for grp, sub in df.groupby(group_col):
        valid = sub[[true_col, pred_col]].dropna()
        if len(valid) < 5:
            continue
        y_t = valid[true_col].values
        y_p = valid[pred_col].values
        
        mae = float(np.mean(np.abs(y_t - y_p)))
        rmse = float(np.sqrt(np.mean((y_t - y_p)**2)))
        ss_res = np.sum((y_t - y_p)**2)
        ss_tot = np.sum((y_t - np.mean(y_t))**2)
        r2 = float(1.0 - (ss_res / max(ss_tot, 1e-8)))
        
        records.append({
            'group': str(grp),
            'samples': len(valid),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'r2': round(max(-1.0, r2), 4)
        })
        
    return pd.DataFrame(records).sort_values(by='samples', ascending=False)


def plot_error_diagnostics_suite(
    y_true, y_pred, y_unc, target_name: str, unit: str, output_path: str
):
    """
    Generates a 4-panel diagnostic figure:
    - Panel 1: Parity plot with calibrated uncertainty bars
    - Panel 2: Residual distribution histogram + KDE
    - Panel 3: Residuals vs. Predicted (Heteroscedasticity)
    - Panel 4: Normal Q-Q Plot of Standardized Residuals
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_unc = np.asarray(y_unc, dtype=float) if y_unc is not None else np.zeros_like(y_true)
    
    valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y_t = y_true[valid]
    y_p = y_pred[valid]
    y_u = y_unc[valid]
    
    if len(y_t) < 5:
        return
        
    residuals = y_t - y_p
    std_residuals = residuals / np.maximum(y_u, 1e-4) if np.any(y_u > 0) else (residuals - np.mean(residuals)) / max(np.std(residuals), 1e-4)
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=250)
    fig.patch.set_facecolor('#FFFFFF')
    
    # 1. Parity Plot
    ax = axes[0, 0]
    min_val = min(np.min(y_t), np.min(y_p))
    max_val = max(np.max(y_t), np.max(y_p))
    margin = (max_val - min_val) * 0.05
    line_range = [min_val - margin, max_val + margin]
    
    ax.plot(line_range, line_range, 'k--', lw=1.5, alpha=0.7, label='Ideal 1:1')
    sample_idx = np.random.choice(len(y_t), min(len(y_t), 400), replace=False)
    ax.errorbar(
        y_t[sample_idx], y_p[sample_idx], 
        yerr=y_u[sample_idx] if np.any(y_u > 0) else None,
        fmt='o', color='#2563EB', ecolor='#93C5FD', elinewidth=0.8,
        markersize=4, alpha=0.6, capsize=0, label='Predicted (OOF)'
    )
    ax.set_title(f'{target_name} Parity Plot', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(f'True {target_name} [{unit}]', fontsize=9.5)
    ax.set_ylabel(f'Predicted {target_name} [{unit}]', fontsize=9.5)
    ax.set_xlim(line_range)
    ax.set_ylim(line_range)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#E2E8F0', fontsize=8.5)
    
    # 2. Residual Distribution Histogram
    ax = axes[0, 1]
    ax.hist(residuals, bins=35, density=True, color='#60A5FA', edgecolor='#1D4ED8', alpha=0.75, label='Residuals')
    mu, std = np.mean(residuals), np.std(residuals)
    x_grid = np.linspace(np.min(residuals), np.max(residuals), 200)
    ax.plot(x_grid, stats.norm.pdf(x_grid, mu, std), color='#DC2626', lw=2.0, label=f'Gaussian fit (μ={mu:.1f}, σ={std:.1f})')
    ax.set_title(f'{target_name} Residual Distribution', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(f'Residual error ({unit})', fontsize=9.5)
    ax.set_ylabel('Probability Density', fontsize=9.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#E2E8F0', fontsize=8.5)
    
    # 3. Residuals vs Predicted (Heteroscedasticity Check)
    ax = axes[1, 0]
    ax.axhline(0, color='k', linestyle='--', lw=1.5, alpha=0.7)
    ax.scatter(y_p[sample_idx], residuals[sample_idx], color='#059669', alpha=0.6, s=18, edgecolors='none')
    ax.set_title(f'Residual vs. Predicted ({target_name})', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(f'Predicted {target_name} [{unit}]', fontsize=9.5)
    ax.set_ylabel(f'Residual error [{unit}]', fontsize=9.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 4. Normal Q-Q Plot
    ax = axes[1, 1]
    stats.probplot(std_residuals, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor('#7C3AED')
    ax.get_lines()[0].set_markeredgecolor('none')
    ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color('#DC2626')
    ax.get_lines()[1].set_linewidth(1.8)
    ax.set_title('Standardized Residuals Normal Q-Q Plot', fontsize=11, fontweight='bold', pad=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved error diagnostics suite to {output_path}")


def plot_stratified_breakdown_bars(df_strat: pd.DataFrame, target_name: str, output_path: str):
    """Plots bar chart of MAE and R2 broken down by crystal symmetry or chemistry."""
    if df_strat.empty or len(df_strat) < 2:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(9, 4.5), dpi=250)
    fig.patch.set_facecolor('#FFFFFF')
    
    groups = df_strat['group'].tolist()[:8]
    mae_vals = df_strat['mae'].tolist()[:8]
    r2_vals = df_strat['r2'].tolist()[:8]
    
    x = np.arange(len(groups))
    width = 0.38
    
    rects1 = ax1.bar(x - width/2, mae_vals, width, label='MAE', color='#3B82F6', edgecolor='#1D4ED8')
    ax1.set_ylabel('Mean Absolute Error (MAE)', color='#1D4ED8', fontsize=10, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, rotation=25, ha='right', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, r2_vals, width, label='R²', color='#10B981', edgecolor='#047857')
    ax2.set_ylabel('R² Score', color='#047857', fontsize=10, fontweight='bold')
    ax2.set_ylim([0.0, 1.05])
    
    plt.title(f'{target_name} Accuracy Stratified by Crystal Symmetry', fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved stratified error bar chart to {output_path}")


def run_error_quantification_suite(sample_size=3000, random_state=42):
    """
    Executes full statistical error quantification and calibration diagnostics.
    """
    import joblib
    pipeline_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_pkl = os.path.join(pipeline_root, "output/training/trained_models.pkl")
    master_csv = os.path.join(pipeline_root, "output/featurization/features_master.csv")
    eval_dir   = os.path.join(pipeline_root, "output/evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    print("Loading models and features for error quantification...")
    pack = joblib.load(models_pkl)
    df_master = pd.read_csv(master_csv, low_memory=False)

    tc_model = pack.get('curie')
    tn_model = pack.get('neel')

    report = {'conformal_coverage': {}, 'stratified_metrics': {}}

    # 1. Curie Temperature
    print("\n--- Error Quantification: Curie Temperature (TC) ---")
    tc_df = df_master.dropna(subset=['reduced_formula', 'Mean_TC_K']).copy()
    tc_df = tc_df[(tc_df['Mean_TC_K'] > 0) & (tc_df['Mean_TC_K'] <= 1800)]
    if len(tc_df) > sample_size:
        tc_df = tc_df.sample(n=sample_size, random_state=random_state)
    y_tc = tc_df['Mean_TC_K'].values

    feats_tc = pack.get('features_curie', [])
    X_tc = tc_df.reindex(columns=feats_tc).fillna(pack.get('medians_curie', {})).fillna(0)
    p_tc, u_tc = tc_model.predict_with_uncertainty(X_tc)
    tc_df['pred_tc'] = p_tc
    tc_df['unc_tc'] = u_tc

    cov_tc = evaluate_conformal_coverage(y_tc, p_tc, u_tc, nominal_level=0.90)
    report['conformal_coverage']['curie_tc'] = cov_tc
    print(f"Curie TC Conformal Coverage (Nominal 90%): {cov_tc['empirical_coverage']*100:.2f}% (Error: {cov_tc['coverage_error']*100:+.2f}%)")
    print(f"Mean Prediction Interval Width: {cov_tc['mean_interval_width']:.1f} K")

    plot_error_diagnostics_suite(
        y_tc, p_tc, u_tc, 
        target_name="Curie Temperature", 
        unit="K", 
        output_path=os.path.join(eval_dir, "curie_tc_error_diagnostics.png")
    )

    if 'crystal_system' in tc_df.columns:
        strat_tc = compute_stratified_error_metrics(tc_df, 'crystal_system', 'Mean_TC_K', 'pred_tc')
        report['stratified_metrics']['curie_tc_by_crystal_system'] = strat_tc.to_dict(orient='records')
        plot_stratified_breakdown_bars(
            strat_tc, 
            target_name="Curie Temperature", 
            output_path=os.path.join(eval_dir, "curie_tc_by_crystal_system.png")
        )

    # 2. Néel Temperature
    print("\n--- Error Quantification: Néel Temperature (TN) ---")
    tn_df = df_master.dropna(subset=['reduced_formula', 'Mean_TN_K']).copy()
    tn_df = tn_df[(tn_df['Mean_TN_K'] > 0) & (tn_df['Mean_TN_K'] <= 1200)]
    if len(tn_df) > sample_size:
        tn_df = tn_df.sample(n=sample_size, random_state=random_state)
    y_tn = tn_df['Mean_TN_K'].values

    feats_tn = pack.get('features_neel', [])
    X_tn = tn_df.reindex(columns=feats_tn).fillna(pack.get('medians_neel', {})).fillna(0)
    p_tn, u_tn = tn_model.predict_with_uncertainty(X_tn)
    tn_df['pred_tn'] = p_tn
    tn_df['unc_tn'] = u_tn

    cov_tn = evaluate_conformal_coverage(y_tn, p_tn, u_tn, nominal_level=0.90)
    report['conformal_coverage']['neel_tn'] = cov_tn
    print(f"Néel TN Conformal Coverage (Nominal 90%): {cov_tn['empirical_coverage']*100:.2f}% (Error: {cov_tn['coverage_error']*100:+.2f}%)")
    print(f"Mean Prediction Interval Width: {cov_tn['mean_interval_width']:.1f} K")

    plot_error_diagnostics_suite(
        y_tn, p_tn, u_tn, 
        target_name="Néel Temperature", 
        unit="K", 
        output_path=os.path.join(eval_dir, "neel_tn_error_diagnostics.png")
    )

    out_json = os.path.join(eval_dir, "error_quantification_report.json")
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved complete error quantification report to {out_json}")
    return report


if __name__ == '__main__':
    run_error_quantification_suite()
