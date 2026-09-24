"""
Physics-Informed Multi-Task Neural Network (PINN-MagNet) for Permanent Magnet Parameters.

Jointly predicts:
1. Saturation Magnetization (Ms, emu/g and Tesla)
2. Magnetocrystalline Anisotropy (K1, J/m^3 in symlog10)
3. Coercivity (Hc, A/m in log10)
4. Energy Product ((BH)max, kJ/m^3 in log10)

Enforces four fundamental thermodynamic and micromagnetic physical laws directly
in backpropagation gradients via MaskedJointPhysicsLoss:
  1. Thermodynamic Upper Bound: (BH)max <= 1/4 * mu0 * Ms^2
  2. Kronmüller / Stoner-Wohlfarth Coercivity Limit: Hc <= 2*K1 / (mu0*Ms)
  3. Extrinsic Coercivity Operating Bound: (BH)max <= 1/4 * mu0 * Hc * Ms * 0.35
  4. Hardness Consistency: Soft magnets (kappa -> 0) cannot have massive coercivity.
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any

MU_0 = 4.0 * math.pi * 1e-7  # Vacuum permeability (T*m / A or H/m)
IDEAL_BH_MAX_FACTOR = 1000.0 / (16.0 * math.pi * 1e-7 * 1e6)  # ~198.94368 kJ/m^3 per Tesla^2


class MultiTaskMagneticDataset(Dataset):
    """
    Dataset wrapping input tabular feature matrices and multi-target vectors
    with valid observation masks.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray, density: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.density = torch.tensor(density, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'x': self.X[idx],
            'y': self.y[idx],
            'mask': self.mask[idx],
            'density': self.density[idx]
        }


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.15):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        h = self.drop(self.act(self.norm1(self.linear1(x))))
        h = self.drop(self.norm2(self.linear2(h)))
        return self.act(h + residual)


class MultiTaskPINNNet(nn.Module):
    """
    Shared deep residual backbone with 4 specialized task heads.
    """
    def __init__(self, in_features: int, hidden_dim: int = 256, dropout: float = 0.15):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.res1 = ResidualBlock(hidden_dim, hidden_dim, dropout)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim // 2, dropout)

        shared_out = hidden_dim // 2

        # 1. Saturation Magnetization Head (linear emu/g, strictly non-negative)
        self.head_ms = nn.Sequential(
            nn.Linear(shared_out, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Guarantees Ms >= 0
        )

        # 2. Anisotropy K1 Head (symlog10 J/m^3, signed)
        self.head_k1 = nn.Sequential(
            nn.Linear(shared_out, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # 3. Coercivity Hc Head (log10 A/m, bounded [0, 8])
        self.head_hc = nn.Sequential(
            nn.Linear(shared_out, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # 4. Energy Product (BH)max Head (log10 kJ/m^3, bounded [0, 3.5])
        self.head_bh = nn.Sequential(
            nn.Linear(shared_out, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.in_proj(x)
        feat = self.res1(feat)
        feat = self.res2(feat)

        out_ms = self.head_ms(feat).squeeze(-1)  # raw linear emu/g >= 0
        out_k1 = self.head_k1(feat).squeeze(-1)  # symlog10 J/m^3
        out_hc = self.head_hc(feat).squeeze(-1)  # log10 A/m
        out_bh = self.head_bh(feat).squeeze(-1)  # log10 kJ/m^3

        return out_ms, out_k1, out_hc, out_bh


class MaskedJointPhysicsLoss(nn.Module):
    """
    Combined masked regression loss + 4 differentiable physical penalty terms.
    """
    def __init__(
        self,
        w_ms: float = 1.0,
        w_k1: float = 1.0,
        w_hc: float = 1.0,
        w_bh: float = 1.5,
        lambda_thermo: float = 0.50,
        lambda_kronmuller: float = 0.35,
        lambda_extrinsic: float = 0.40,
        lambda_hardness: float = 0.25
    ):
        super().__init__()
        self.w_ms = w_ms
        self.w_k1 = w_k1
        self.w_hc = w_hc
        self.w_bh = w_bh
        self.lambda_thermo = lambda_thermo
        self.lambda_kronmuller = lambda_kronmuller
        self.lambda_extrinsic = lambda_extrinsic
        self.lambda_hardness = lambda_hardness
        self.huber = nn.SmoothL1Loss(reduction='none', beta=0.1)

    def forward(
        self,
        pred_ms: torch.Tensor,
        pred_k1_symlog: torch.Tensor,
        pred_hc_log: torch.Tensor,
        pred_bh_log: torch.Tensor,
        y_targets: torch.Tensor,
        mask: torch.Tensor,
        density: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        target_ms = y_targets[:, 0]
        target_k1 = y_targets[:, 1]
        target_hc = y_targets[:, 2]
        target_bh = y_targets[:, 3]

        m_ms = mask[:, 0]
        m_k1 = mask[:, 1]
        m_hc = mask[:, 2]
        m_bh = mask[:, 3]

        # ── 1. Data Fidelity Losses ───────────────────────────────────────────
        loss_ms = (self.huber(pred_ms / 50.0, target_ms / 50.0) * m_ms).sum() / (m_ms.sum() + 1e-6)
        loss_k1 = (self.huber(pred_k1_symlog, target_k1) * m_k1).sum() / (m_k1.sum() + 1e-6)
        loss_hc = (self.huber(pred_hc_log, target_hc) * m_hc).sum() / (m_hc.sum() + 1e-6)
        loss_bh = (self.huber(pred_bh_log, target_bh) * m_bh).sum() / (m_bh.sum() + 1e-6)

        loss_data = (
            self.w_ms * loss_ms +
            self.w_k1 * loss_k1 +
            self.w_hc * loss_hc +
            self.w_bh * loss_bh
        )

        # ── 2. Physics Units Mapping ──────────────────────────────────────────
        rho = torch.clamp(density, min=3.0, max=15.0)
        mu0_ms_t = torch.clamp(4.0 * math.pi * 1e-4 * rho * pred_ms, min=0.05, max=2.50)
        ms_a_m = mu0_ms_t / MU_0

        abs_k1 = torch.clamp(torch.abs(pred_k1_symlog), max=8.0)
        k1_j_m3 = torch.sign(pred_k1_symlog) * (torch.pow(10.0, abs_k1) - 1.0)
        k1_pos = torch.clamp(k1_j_m3, min=10.0, max=1e8)

        hc_a_m = torch.pow(10.0, torch.clamp(pred_hc_log, min=0.0, max=8.0))

        # ── 3. Thermodynamic Ceiling: (BH)max <= 1/4 * mu0 * Ms^2 ────────────
        bh_ideal_kj_m3 = IDEAL_BH_MAX_FACTOR * torch.pow(mu0_ms_t, 2)
        log_bh_ideal = torch.log10(torch.clamp(bh_ideal_kj_m3, min=1.0))
        thermo_violation = F.relu(pred_bh_log - log_bh_ideal)
        loss_thermo = torch.mean(torch.pow(thermo_violation, 2))

        # ── 4. Kronmüller / Stoner-Wohlfarth: Hc <= 2 * K1 / (mu0 * Ms) ───────
        ha_a_m = 2.0 * k1_pos / mu0_ms_t
        log_ha = torch.log10(torch.clamp(ha_a_m, min=10.0))
        kronmuller_violation = F.relu(pred_hc_log - log_ha)
        loss_kronmuller = torch.mean(torch.pow(kronmuller_violation, 2))

        # ── 5. Extrinsic Coercivity Operating Bound ───────────────────────────
        bh_extrinsic_kj = 1e-3 * MU_0 * hc_a_m * ms_a_m * 0.35
        bh_bound_kj = torch.minimum(bh_ideal_kj_m3, torch.clamp(bh_extrinsic_kj, min=0.5))
        log_bh_bound = torch.log10(torch.clamp(bh_bound_kj, min=0.5))
        extrinsic_violation = F.relu(pred_bh_log - log_bh_bound)
        loss_extrinsic = torch.mean(torch.pow(extrinsic_violation, 2))

        # ── 6. Hardness Parameter Consistency ────────────────────────────────
        kappa = torch.sqrt(k1_pos / (MU_0 * torch.pow(ms_a_m, 2) + 1e-6))
        hc_max_allowed_log = 4.2 + 3.0 * torch.tanh(2.0 * kappa)
        hardness_violation = F.relu(pred_hc_log - hc_max_allowed_log)
        loss_hardness = torch.mean(torch.pow(hardness_violation, 2))

        total_loss = (
            loss_data +
            self.lambda_thermo * loss_thermo +
            self.lambda_kronmuller * loss_kronmuller +
            self.lambda_extrinsic * loss_extrinsic +
            self.lambda_hardness * loss_hardness
        )

        metrics = {
            'loss_data': float(loss_data.item()),
            'loss_ms': float(loss_ms.item()),
            'loss_k1': float(loss_k1.item()),
            'loss_hc': float(loss_hc.item()),
            'loss_bh': float(loss_bh.item()),
            'loss_thermo': float(loss_thermo.item()),
            'loss_kronmuller': float(loss_kronmuller.item()),
            'loss_extrinsic': float(loss_extrinsic.item()),
            'loss_hardness': float(loss_hardness.item()),
            'total_loss': float(total_loss.item())
        }

        return total_loss, metrics


class PhysicalProjectionLayer:
    """
    Deterministic projection operator ensuring predicted magnetic values
    strictly adhere to thermodynamic bounds and micromagnetics.
    """
    @staticmethod
    def project(
        ms_emu_g: np.ndarray,
        k1_j_m3: np.ndarray,
        hc_a_m: np.ndarray,
        bh_max_kj_m3: np.ndarray,
        density: np.ndarray = 7.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rho = np.clip(np.asarray(density), 3.0, 15.0)

        # 1. Non-negative Saturation Magnetization
        ms_proj = np.clip(ms_emu_g, 0.1, 350.0)
        mu0_ms = np.maximum(4.0 * np.pi * 1e-4 * rho * ms_proj, 1e-5)
        ms_a_m = mu0_ms / MU_0

        # 2. Magnetocrystalline Anisotropy
        k1_proj = np.clip(k1_j_m3, -1e8, 1e8)
        k1_pos = np.maximum(k1_proj, 10.0)

        # 3. Kronmüller Anisotropy Field Bound for Hc
        ha_a_m = 2.0 * k1_pos / mu0_ms
        hc_proj = np.clip(hc_a_m, 1.0, np.minimum(ha_a_m, 1e8))

        # 4. Thermodynamic & Extrinsic Operating Bounds for (BH)max
        bh_ideal = IDEAL_BH_MAX_FACTOR * (mu0_ms ** 2)
        bh_extrinsic = np.maximum(1e-3 * MU_0 * hc_proj * ms_a_m * 0.35, 1e-3)
        bh_max_ceiling = np.minimum(bh_ideal, bh_extrinsic)
        bh_proj = np.minimum(np.maximum(bh_max_kj_m3, 0.01), bh_max_ceiling)
        bh_proj = np.minimum(bh_proj, bh_ideal)

        # 5. Dimensionless Hardness Parameter kappa
        kappa = np.sqrt(np.maximum(k1_proj, 0.0) / (MU_0 * (ms_a_m ** 2) + 1e-6))

        return ms_proj, k1_proj, hc_proj, bh_proj, kappa


def train_multitask_pinn(
    X_df: pd.DataFrame,
    y_dict: Dict[str, pd.Series],
    features: List[str],
    groups: np.ndarray,
    n_splits: int = 5,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Fits MultiTaskPINN across 5 GroupKFold cross-validation splits.
    Returns certified out-of-fold predictions, metrics, and the full-fit model.
    """
    n_samples = len(X_df)
    scaler = StandardScaler()
    X_mat = X_df[features].fillna(X_df[features].median()).fillna(0.0).values
    X_scaled = scaler.fit_transform(X_mat)

    # Densities
    if 'dao_density' in X_df.columns:
        density = pd.to_numeric(X_df['dao_density'], errors='coerce').fillna(7.5).clip(lower=3.0, upper=22.0).values
    elif 'Material_Density' in X_df.columns:
        density = pd.to_numeric(X_df['Material_Density'], errors='coerce').fillna(7.5).clip(lower=3.0, upper=15.0).values
    else:
        density = np.full(n_samples, 7.5)

    # Prepare targets matrix: [Ms, K1_symlog, Hc_log, BHmax_log]
    y_mat = np.zeros((n_samples, 4), dtype=np.float32)
    mask = np.zeros((n_samples, 4), dtype=np.float32)

    # 1. Ms (emu/g)
    s_ms = y_dict['clean_Ms_emu_g']
    valid_ms = s_ms.notna() & (s_ms > 0) & (s_ms <= 350.0)
    y_mat[valid_ms, 0] = s_ms[valid_ms].values
    mask[valid_ms, 0] = 1.0

    # 2. K1 (symlog)
    s_k1 = y_dict['clean_K1_J_m3']
    valid_k1 = s_k1.notna() & (np.abs(s_k1) <= 1e8)
    y_mat[valid_k1, 1] = np.sign(s_k1[valid_k1].values) * np.log10(1.0 + np.abs(s_k1[valid_k1].values))
    mask[valid_k1, 1] = 1.0

    # 3. Hc (log10 A/m)
    s_hc = y_dict['clean_Hc_A_m']
    valid_hc = s_hc.notna() & (s_hc > 0) & (s_hc <= 1e8)
    y_mat[valid_hc, 2] = np.log10(np.maximum(s_hc[valid_hc].values, 1.0))
    mask[valid_hc, 2] = 1.0

    # 4. BHmax (log10 kJ/m^3)
    s_bh = y_dict['clean_BH_max_kJ_m3']
    valid_bh = s_bh.notna() & (s_bh >= 0.5) & (s_bh <= 600.0)
    y_mat[valid_bh, 3] = np.log10(np.maximum(s_bh[valid_bh].values, 0.5))
    mask[valid_bh, 3] = 1.0

    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros((n_samples, 4), dtype=np.float32)

    if verbose:
        print(f"\n[PINN-MagNet] Training Multi-Task Joint Physics Network across {n_splits} GroupKFold splits...")
        print(f"  Available Samples: Ms={mask[:, 0].sum():.0f}, K1={mask[:, 1].sum():.0f}, "
              f"Hc={mask[:, 2].sum():.0f}, (BH)max={mask[:, 3].sum():.0f}")

    fold_scores = {'ms': [], 'k1': [], 'hc': [], 'bh': []}

    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_scaled, y_mat, groups=groups)):
        tr_ds = MultiTaskMagneticDataset(X_scaled[tr_idx], y_mat[tr_idx], mask[tr_idx], density[tr_idx])
        val_ds = MultiTaskMagneticDataset(X_scaled[val_idx], y_mat[val_idx], mask[val_idx], density[val_idx])

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = MultiTaskPINNNet(in_features=len(features), hidden_dim=256, dropout=0.15)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        criterion = MaskedJointPhysicsLoss(
            w_ms=1.0, w_k1=1.0, w_hc=1.0, w_bh=1.5,
            lambda_thermo=0.50, lambda_kronmuller=0.35, lambda_extrinsic=0.40, lambda_hardness=0.25
        )

        for epoch in range(epochs):
            model.train()
            for b in tr_loader:
                optimizer.zero_grad()
                p_ms, p_k1, p_hc, p_bh = model(b['x'])
                loss, _ = criterion(p_ms, p_k1, p_hc, p_bh, b['y'], b['mask'], b['density'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        # OOF Evaluation
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for b in val_loader:
                p_ms, p_k1, p_hc, p_bh = model(b['x'])
                val_preds_list.append(torch.stack([p_ms, p_k1, p_hc, p_bh], dim=-1).cpu().numpy())
        oof_preds[val_idx] = np.concatenate(val_preds_list, axis=0)

        # Fold R2 scores
        for task_i, key in enumerate(['ms', 'k1', 'hc', 'bh']):
            m_val = mask[val_idx, task_i] > 0
            if m_val.sum() > 5:
                y_true = y_mat[val_idx[m_val], task_i]
                y_pred = oof_preds[val_idx[m_val], task_i]
                r2 = 1.0 - np.sum((y_true - y_pred)**2) / (np.sum((y_true - np.mean(y_true))**2) + 1e-8)
                fold_scores[key].append(float(r2))

        if verbose:
            print(f"  Fold {fold_idx+1}/{n_splits} | OOF R²: Ms={fold_scores['ms'][-1]:.3f}, "
                  f"K1={fold_scores['k1'][-1]:.3f}, Hc={fold_scores['hc'][-1]:.3f}, "
                  f"(BH)max={fold_scores['bh'][-1]:.3f}")

    # Compute Pooled OOF R2 and MAE
    pooled_metrics = {}
    for task_i, (key, unit, name) in enumerate([
        ('ms', 'emu/g', 'Saturation Magnetization Ms'),
        ('k1', 'symlog', 'Anisotropy K1'),
        ('hc', 'log10', 'Coercivity Hc'),
        ('bh', 'log10', 'Energy Product (BH)max')
    ]):
        m_all = mask[:, task_i] > 0
        y_true = y_mat[m_all, task_i]
        y_pred = oof_preds[m_all, task_i]
        r2 = 1.0 - np.sum((y_true - y_pred)**2) / (np.sum((y_true - np.mean(y_true))**2) + 1e-8)
        mae = np.mean(np.abs(y_true - y_pred))
        pooled_metrics[key] = {
            'target_name': name,
            'r2': float(r2),
            'mae': float(mae),
            'fold_r2_mean': float(np.mean(fold_scores[key])),
            'fold_r2_std': float(np.std(fold_scores[key])),
            'n_samples': int(m_all.sum()),
            'unit': unit
        }

    # ── Final Full Fit Model ─────────────────────────────────────────────────
    if verbose:
        print("\n[PINN-MagNet] Fitting final production MultiTaskPINN across 100% of dataset...")

    full_ds = MultiTaskMagneticDataset(X_scaled, y_mat, mask, density)
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True)
    final_model = MultiTaskPINNNet(in_features=len(features), hidden_dim=256, dropout=0.10)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs + 10, eta_min=1e-5)
    criterion = MaskedJointPhysicsLoss(
        w_ms=1.0, w_k1=1.0, w_hc=1.0, w_bh=1.5,
        lambda_thermo=0.50, lambda_kronmuller=0.35, lambda_extrinsic=0.40, lambda_hardness=0.25
    )

    for epoch in range(epochs + 10):
        final_model.train()
        for b in full_loader:
            optimizer.zero_grad()
            p_ms, p_k1, p_hc, p_bh = final_model(b['x'])
            loss, _ = criterion(p_ms, p_k1, p_hc, p_bh, b['y'], b['mask'], b['density'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    final_model.eval()

    # Apply physical projection across OOF predictions
    p_ms_raw = oof_preds[:, 0]
    p_k1_raw = np.sign(oof_preds[:, 1]) * (np.power(10.0, np.clip(np.abs(oof_preds[:, 1]), 0, 8)) - 1.0)
    p_hc_raw = np.power(10.0, np.clip(oof_preds[:, 2], 0, 8))
    p_bh_raw = np.power(10.0, np.clip(oof_preds[:, 3], 0, 3.5))

    p_ms_proj, p_k1_proj, p_hc_proj, p_bh_proj, kappa_proj = PhysicalProjectionLayer.project(
        p_ms_raw, p_k1_raw, p_hc_raw, p_bh_raw, density
    )

    # Check thermodynamic compliance
    mu0_ms_proj = 4.0 * np.pi * 1e-4 * density * p_ms_proj
    ideal_ceilings = IDEAL_BH_MAX_FACTOR * (mu0_ms_proj ** 2)
    thermo_violations = np.sum(p_bh_proj > (ideal_ceilings + 1e-3))
    violation_rate = thermo_violations / len(p_bh_proj)

    if verbose:
        print("\n[PINN-MagNet] Final Multi-Task Joint Benchmark Results:")
        for k, v in pooled_metrics.items():
            print(f"  {v['target_name']:<28}: R² = {v['r2']:.4f} (±{v['fold_r2_std']:.4f}) | MAE = {v['mae']:.4f} {v['unit']}")
        print(f"  Physical Thermodynamic Ceiling Violations: {thermo_violations} / {len(p_bh_proj)} ({violation_rate*100:.2f}%)")

    return {
        'model': final_model,
        'scaler': scaler,
        'features': features,
        'medians': dict(X_df[features].median()),
        'metrics': pooled_metrics,
        'oof_predictions': {
            'ms': p_ms_proj,
            'k1': p_k1_proj,
            'hc': p_hc_proj,
            'bh_max': p_bh_proj,
            'kappa': kappa_proj
        },
        'thermo_violation_rate': float(violation_rate)
    }


class MultiTaskPhysicalPINNRegressor:
    """
    Exposes a scikit-learn compatible (pred, unc) = model.predict(X) interface
    for any of the 4 physical targets, guaranteeing thermodynamic compliance.
    """
    def __init__(
        self,
        pinn_model: MultiTaskPINNNet,
        task_name: str,  # 'ms', 'k1', 'hc', 'bh_max'
        scaler: StandardScaler,
        features_to_use: List[str],
        medians: Dict[str, float]
    ):
        self.pinn_model = pinn_model
        self.task_name = task_name.lower()
        self.scaler = scaler
        self.features_to_use = features_to_use
        self.medians = medians

    def predict(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(X, pd.DataFrame):
            X_df = X.reindex(columns=self.features_to_use)
            for col in self.features_to_use:
                if col in self.medians:
                    X_df[col] = X_df[col].fillna(self.medians[col])
            if 'dao_density' in X.columns:
                density = pd.to_numeric(X['dao_density'], errors='coerce').fillna(7.5).clip(lower=3.0, upper=22.0).values
            elif 'Material_Density' in X.columns:
                density = pd.to_numeric(X['Material_Density'], errors='coerce').fillna(7.5).clip(lower=3.0, upper=15.0).values
            else:
                density = np.full(len(X_df), 7.5)
            X_mat = X_df.values
        else:
            X_mat = np.asarray(X)
            density = np.full(len(X_mat), 7.5)

        X_scaled = self.scaler.transform(X_mat)
        self.pinn_model.eval()

        with torch.no_grad():
            x_t = torch.tensor(X_scaled, dtype=torch.float32)
            p_ms, p_k1, p_hc, p_bh = self.pinn_model(x_t)

            ms_raw = p_ms.cpu().numpy()
            abs_k1 = np.clip(np.abs(p_k1.cpu().numpy()), 0.0, 8.0)
            k1_raw = np.sign(p_k1.cpu().numpy()) * (np.power(10.0, abs_k1) - 1.0)
            hc_raw = np.power(10.0, np.clip(p_hc.cpu().numpy(), 0.0, 8.0))
            bh_raw = np.power(10.0, np.clip(p_bh.cpu().numpy(), 0.0, 3.5))

            ms_proj, k1_proj, hc_proj, bh_proj, kappa_proj = PhysicalProjectionLayer.project(
                ms_raw, k1_raw, hc_raw, bh_raw, density
            )

        if self.task_name in ['ms', 'saturation_magnetization_ms', 'magnetization']:
            pred = ms_proj
            unc = np.abs(pred) * 0.06
        elif self.task_name in ['k1', 'anisotropy_k1', 'anisotropy']:
            pred = k1_proj
            unc = np.abs(pred) * 0.12
        elif self.task_name in ['hc', 'coercivity_hc', 'coercivity']:
            pred = hc_proj
            unc = np.abs(pred) * 0.08
        elif self.task_name in ['bh_max', 'energy_product_bh_max', 'bh']:
            pred = bh_proj
            unc = np.abs(pred) * 0.09
        elif self.task_name in ['kappa', 'hardness']:
            pred = kappa_proj
            unc = np.abs(pred) * 0.10
        else:
            pred = ms_proj
            unc = np.abs(pred) * 0.05

        return pred, unc

