#!/usr/bin/env python3
"""
================================================================================
PHYSICS-INFORMED MACHINE LEARNING PIPELINE FOR MAGNETIC MATERIALS DISCOVERY
================================================================================
A unified, production-grade end-to-end framework integrating:
  - Stage 1: Data ingestion & crystallographic mapping (NEMAD + Materials Project)
  - Stage 2: Physical featurization & leakage-safe feature selection
  - Stage 3: Exploratory data analysis & publication visualizations
  - Stage 4: Specialist-corrected multi-task classification & non-negative regression
  - Stage 5: Calibration & soft probabilistic inference on unseen candidate pools
  - Stage 6: Rare-earth-free sustainable permanent magnet discovery screening
"""

import os
import sys
import time
import argparse

# Ensure ML_pipeline root is in sys.path and resolve directories
PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from src.preprocessing import preprocess_all_datasets
from src.featurization import featurize_all_datasets
from src.visualization import run_all_eda
from src.modeling import run_all_modeling
from src.post_processing import run_magnetic_interpretation, run_discovery_screening
from src.generative_screening import GenerativeCrystalScreeningEngine

def print_header(title):
    print("\n" + "=" * 76)
    print(f"  {title.center(72)}")
    print("=" * 76)

def run_single_prediction(formula, models_dir=None):
    """Predict magnetic phase, ordering temperature, and CRM rating for a formula."""
    print_header(f"PREDICTING PROPERTIES FOR: {formula}")
    models_dir = models_dir or os.path.join(PIPELINE_ROOT, 'output', 'training')
    models_path = os.path.join(models_dir, 'trained_models.pkl')
    
    if not os.path.exists(models_path):
        print(f"Error: Trained models not found at {models_path}. Run Stage 4 first.")
        return None
        
    engine = GenerativeCrystalScreeningEngine(models_path=models_path)
    result = engine.screen_crystal(formula)
    
    print("\n--- Model Predictions ---")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Unified Physics-Informed Magnetic Materials Discovery Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--stage',
        choices=['1', '2', '3', '4', '5', '6', 'all'],
        default='all',
        help="Pipeline stage to execute (1: Preprocess, 2: Featurize, 3: EDA, 4: Train, 5: Predict, 6: Discover, all: Full Pipeline)"
    )
    parser.add_argument(
        '--predict',
        type=str,
        default=None,
        help="Quick inference for a single chemical formula (e.g. --predict 'Fe3Sn' or 'Nd2Fe14B')"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=os.path.join(PIPELINE_ROOT, 'output'),
        help="Base output directory (default: ML_pipeline/output)"
    )
    parser.add_argument(
        '--plots-dir',
        type=str,
        default=os.path.join(PIPELINE_ROOT, 'plots'),
        help="Directory to save generated figures (default: ML_pipeline/plots)"
    )

    args = parser.parse_args()

    # Handle quick prediction mode
    if args.predict:
        run_single_prediction(args.predict, models_dir=os.path.join(args.output_dir, 'training'))
        return

    print_header("PHYSICS-INFORMED INTERPRETABLE MAGNETIC MATERIALS PIPELINE")
    start_time = time.time()

    preprocessed_dir = os.path.join(args.output_dir, 'preprocessing')
    featurized_dir = os.path.join(args.output_dir, 'featurization')
    training_dir = os.path.join(args.output_dir, 'training')
    post_processing_dir = os.path.join(args.output_dir, 'post_processing')
    plot_dir = args.plots_dir

    stages_to_run = [args.stage] if args.stage != 'all' else ['1', '2', '3', '4', '5', '6']

    # Stage 1: Data Preprocessing & Crystallographic Mapping
    if '1' in stages_to_run:
        print_header("STAGE 1: DATA PREPROCESSING & CRYSTALLOGRAPHIC MERGING")
        t0 = time.time()
        preprocess_all_datasets(
            data_dir=os.path.join(PIPELINE_ROOT, 'Dataset'),
            output_dir=preprocessed_dir
        )
        print(f"Stage 1 completed in {time.time() - t0:.2f} seconds.")

    # Stage 2: Physical Featurization & Feature Selection
    if '2' in stages_to_run:
        print_header("STAGE 2: PHYSICAL FEATURIZATION & TARGET-WISE SELECTION")
        t1 = time.time()
        featurize_all_datasets(
            input_dir=preprocessed_dir,
            output_dir=featurized_dir,
            plot_dir=plot_dir,
            make_tsne=True
        )
        print(f"Stage 2 completed in {time.time() - t1:.2f} seconds.")

    # Stage 3: Exploratory Data Analysis & Plots
    if '3' in stages_to_run:
        print_header("STAGE 3: EXPLORATORY DATA ANALYSIS (PHYSICAL PLOTS)")
        t2 = time.time()
        run_all_eda(
            output_dir=preprocessed_dir,
            feat_dir=featurized_dir,
            plot_dir=plot_dir
        )
        print(f"Stage 3 completed in {time.time() - t2:.2f} seconds.")

    # Stage 4: Interpretable Machine Learning Modeling
    if '4' in stages_to_run:
        print_header("STAGE 4: INTERPRETABLE MACHINE LEARNING MODELING")
        run_all_modeling(
            input_dir=featurized_dir,
            output_dir=training_dir,
            plot_dir=plot_dir
        )
        from src.visualization import run_training_visualizations
        run_training_visualizations(pred_dir=training_dir, plot_dir=plot_dir)

    # Stage 5: Inference & Diagnostics on Candidate Pool
    if '5' in stages_to_run:
        print_header("STAGE 5: MAGNETIC INFERENCE ON UNSEEN CANDIDATES")
        t4 = time.time()
        run_magnetic_interpretation(
            output_dir=post_processing_dir,
            training_dir=training_dir,
            preprocessed_dir=preprocessed_dir
        )
        print(f"Stage 5 completed in {time.time() - t4:.2f} seconds.")

    # Stage 6: Sustainable Magnet Discovery Screening
    if '6' in stages_to_run:
        print_header("STAGE 6: SUSTAINABLE MAGNET DISCOVERY SCREENING")
        t5 = time.time()
        run_discovery_screening(
            output_dir=post_processing_dir,
            training_dir=training_dir,
            preprocessed_dir=preprocessed_dir,
            plot_dir=plot_dir
        )
        from src.visualization import run_discovery_visualizations
        run_discovery_visualizations(post_dir=post_processing_dir, plot_dir=plot_dir)
        print(f"Stage 6 completed in {time.time() - t5:.2f} seconds.")

    total_time = time.time() - start_time
    print_header(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time/60:.2f} MINUTES")

if __name__ == '__main__':
    main()
