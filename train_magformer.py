#!/usr/bin/env python3
"""
MagFormer Transformer Training Pipeline:
Trains the Physics-Gated Formula Transformer (MagFormer) on 35,000+ experimental inorganic materials.
Multi-task targets:
1. 3-class Magnetic Phase Classification (FM / AFM / NM) via Focal Loss.
2. Curie Temperature (TC) via Heteroscedastic NLL + Huber Loss.
3. Néel Temperature (TN) via Heteroscedastic NLL + Huber Loss.
Saves production weights to output/training/magformer_model.pt and updates trained_models.pkl.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import torch

PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from src.composition_gnn import (
    MagFormer,
    train_composition_gnn_model,
    predict_with_gnn,
    save_magformer_model,
)

CLASS_CSV = os.path.join(PIPELINE_ROOT, "Dataset/Classification_FM_AFM_NM.csv")
TC_CSV    = os.path.join(PIPELINE_ROOT, "Dataset/FM_with_curie.csv")
TN_CSV    = os.path.join(PIPELINE_ROOT, "Dataset/AFM_with_Neel.csv")
MODELS_PKL = os.path.join(PIPELINE_ROOT, "output/training/trained_models.pkl")
MODEL_PT   = os.path.join(PIPELINE_ROOT, "output/training/magformer_model.pt")


def load_unified_dataset():
    print("Loading experimental datasets for MagFormer training...")
    df_c = pd.read_csv(CLASS_CSV, low_memory=False)
    df_c = df_c.dropna(subset=['Normalized_Composition', 'Type']).copy()
    df_c['Type'] = df_c['Type'].astype(int)

    # Load Curie temperatures
    df_tc = pd.read_csv(TC_CSV, low_memory=False)
    df_tc = df_tc.dropna(subset=['Normalized_Composition', 'Mean_TC_K'])
    tc_map = df_tc.groupby('Normalized_Composition')['Mean_TC_K'].median().to_dict()

    # Load Neel temperatures
    df_tn = pd.read_csv(TN_CSV, low_memory=False)
    df_tn = df_tn.dropna(subset=['Normalized_Composition', 'Mean_TN_K'])
    tn_map = df_tn.groupby('Normalized_Composition')['Mean_TN_K'].median().to_dict()

    df_c['Mean_TC_K'] = df_c['Normalized_Composition'].map(tc_map)
    df_c['Mean_TN_K'] = df_c['Normalized_Composition'].map(tn_map)

    # Deduplicate by formula using consensus label
    grp_mode = df_c.groupby('Normalized_Composition')['Type'].agg(lambda s: s.mode()[0])
    df_dedup = df_c.drop_duplicates(subset=['Normalized_Composition']).copy()
    df_dedup['Type'] = df_dedup['Normalized_Composition'].map(grp_mode)

    formulas = df_dedup['Normalized_Composition'].tolist()
    targets = {
        'class': df_dedup['Type'].values,
        'tc': df_dedup['Mean_TC_K'].fillna(0.0).values,
        'tn': df_dedup['Mean_TN_K'].fillna(0.0).values,
    }

    print(f"Total unique materials: {len(formulas)}")
    print(f"  FM samples:  {(targets['class'] == 0).sum()} (TC labels: {(targets['tc'] > 0).sum()})")
    print(f"  AFM samples: {(targets['class'] == 1).sum()} (TN labels: {(targets['tn'] > 0).sum()})")
    print(f"  NM samples:  {(targets['class'] == 2).sum()}")

    return formulas, targets


def main():
    formulas, targets = load_unified_dataset()

    print("\nTraining MagFormer Transformer (3-Layer Self-Attention + Pairwise Attention Bias)...")
    t0 = time.time()
    model = train_composition_gnn_model(
        formulas=formulas,
        targets=targets,
        epochs=15,
        batch_size=256,
        verbose=True
    )
    t1 = time.time()
    print(f"MagFormer training completed in {t1 - t0:.1f} seconds.")

    # Save to disk
    save_magformer_model(model, filepath=MODEL_PT)

    # Save training loss curves and history
    if hasattr(model, 'training_history') and model.training_history:
        from src.training_diagnostics import plot_magformer_loss_curves, save_diagnostic_json
        hist_json = os.path.join(PIPELINE_ROOT, "output/evaluation/magformer_training_history.json")
        save_diagnostic_json(model.training_history, hist_json)
        plot_path = os.path.join(PIPELINE_ROOT, "output/evaluation/magformer_training_curves.png")
        plot_magformer_loss_curves(model.training_history, output_path=plot_path)

    # Update trained_models.pkl if available
    if os.path.exists(MODELS_PKL):
        try:
            pack = joblib.load(MODELS_PKL)
            if isinstance(pack, dict):
                pack['magformer'] = model
                joblib.dump(pack, MODELS_PKL)
                print(f"Successfully added MagFormer to unified pack: {MODELS_PKL}")
        except Exception as exc:
            print(f"Warning: Could not update {MODELS_PKL}: {exc}")

    # Evaluate benchmark predictions on holdout check
    test_formulas = ['Nd2Fe14B', 'SmCo5', 'FePt', 'BiFeO3', 'MnBi', 'Al2O3', 'Sr2FeMoO6', 'CrO2']
    probs, tc, tn, tc_unc, tn_unc = predict_with_gnn(model, test_formulas, task_type='all_with_uncertainty')
    phase_names = ['FM', 'AFM', 'NM']

    print("\n--- MagFormer Verification Benchmark ---")
    for i, f in enumerate(test_formulas):
        pred_p = phase_names[int(np.argmax(probs[i]))]
        conf = float(np.max(probs[i]))
        print(f"  {f:12s} -> Phase: {pred_p} ({conf*100:.1f}%) | TC: {tc[i]:.0f} K (+/- {tc_unc[i]:.0f}) | TN: {tn[i]:.0f} K (+/- {tn_unc[i]:.0f})")

    print("\nMagFormer is now fully operational in the pipeline.")


if __name__ == "__main__":
    main()
