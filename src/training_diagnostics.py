#!/usr/bin/env python3
"""
Training Diagnostics & Loss Curve Plotting Module.
Captures and visualizes convergence trajectories for:
1. MagFormer Transformer (Multi-task Focal Loss + Heteroscedastic NLL across epochs)
2. Tabular GBDT Ensembles (CatBoost, LightGBM, XGBoost train vs. validation loss across iterations)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#D1D5DB'
plt.rcParams['axes.linewidth'] = 0.8


def plot_magformer_loss_curves(history: dict, output_path: str = "output/evaluation/magformer_training_curves.png"):
    """
    Plots multi-task training loss trajectories for MagFormer across epochs:
    - Panel 1: Total Multi-Task Loss
    - Panel 2: 3-Class Focal Loss
    - Panel 3: Curie (TC) and Néel (TN) Heteroscedastic NLL
    - Panel 4: Learning Rate Schedule
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = history.get('epoch', list(range(1, len(history.get('loss_total', [])) + 1)))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=250)
    fig.patch.set_facecolor('#FFFFFF')
    
    # 1. Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['loss_total'], color='#1E40AF', lw=2.2, marker='o', markersize=4, label='Total Loss')
    ax.set_title('Total Multi-Task Loss', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('Epoch', fontsize=9.5)
    ax.set_ylabel('Loss', fontsize=9.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#E2E8F0', fontsize=9)
    
    # 2. Classification Focal Loss
    ax = axes[0, 1]
    ax.plot(epochs, history['loss_class'], color='#D97706', lw=2.2, marker='s', markersize=4, label='Focal Loss (Phase)')
    ax.set_title('Magnetic Phase Classification Loss', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('Epoch', fontsize=9.5)
    ax.set_ylabel('Focal Loss', fontsize=9.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#E2E8F0', fontsize=9)
    
    # 3. Transition Temperatures Heteroscedastic NLL
    ax = axes[1, 0]
    ax.plot(epochs, history['loss_tc'], color='#DC2626', lw=2.0, marker='^', markersize=4, label='Curie TC (NLL + Huber)')
    ax.plot(epochs, history['loss_tn'], color='#059669', lw=2.0, marker='v', markersize=4, label='Néel TN (NLL + Huber)')
    ax.set_title('Transition Temperature Heteroscedastic Losses', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('Epoch', fontsize=9.5)
    ax.set_ylabel('Heteroscedastic Loss', fontsize=9.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#E2E8F0', fontsize=9)
    
    # 4. Learning Rate Schedule
    ax = axes[1, 1]
    ax.plot(epochs, history['lr'], color='#7C3AED', lw=2.0, linestyle='-', label='AdamW Learning Rate')
    ax.set_title('Learning Rate Dynamics', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('Epoch', fontsize=9.5)
    ax.set_ylabel('Learning Rate', fontsize=9.5)
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#E2E8F0', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved MagFormer loss curves to {output_path}")


def plot_gbdt_convergence_curves(
    evals_data: dict, 
    target_name: str, 
    output_path: str
):
    """
    Plots train vs. validation learning curves for CatBoost, LightGBM, and XGBoost across boosting rounds.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    models = [m for m in ['CatBoost', 'LightGBM', 'XGBoost'] if m in evals_data]
    if not models:
        return
        
    n_cols = len(models)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 4.2), dpi=250)
    if n_cols == 1:
        axes = [axes]
    fig.patch.set_facecolor('#FFFFFF')
    
    colors = {
        'CatBoost': ('#0284C7', '#BAE6FD'),
        'LightGBM': ('#16A34A', '#BBF7D0'),
        'XGBoost': ('#EA580C', '#FED7AA')
    }
    
    for ax, m_name in zip(axes, models):
        m_dict = evals_data[m_name]
        train_loss = m_dict.get('train', [])
        val_loss = m_dict.get('val', [])
        iters = list(range(1, len(train_loss) + 1))
        
        main_c, fill_c = colors.get(m_name, ('#4B5563', '#E5E7EB'))
        
        if train_loss:
            ax.plot(iters, train_loss, color=main_c, linestyle='--', lw=1.8, label=f'{m_name} Train')
        if val_loss:
            val_iters = list(range(1, len(val_loss) + 1))
            ax.plot(val_iters, val_loss, color=main_c, linestyle='-', lw=2.2, label=f'{m_name} Validation (OOF)')
            
        ax.set_title(f'{m_name} Convergence ({target_name})', fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Boosting Iterations', fontsize=9.5)
        ax.set_ylabel('Objective Loss (RMSE / NLL)', fontsize=9.5)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#E2E8F0', fontsize=8.5)
        
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved GBDT convergence curves for {target_name} to {output_path}")


def save_diagnostic_json(data: dict, filepath: str):
    """Saves structured metrics or loss history to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved diagnostic record to {filepath}")
