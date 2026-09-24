import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pymatgen.core import Composition, Element

# Set random seed and CPU threads for reproducibility and speed
torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(12)

# Shared magnetic constants imported from central utils
from src.utils import (
    ELEMENTAL_FM,
    ELEMENTAL_AFM,
    STRONG_FM_3D,
    STRONG_AFM_3D,
    RARE_EARTH_FM,
    RARE_EARTH_AFM,
    COMPLEX_RARE_EARTH_MAGNETS,
    MOLECULAR_AFM_ELEMENTS,
    ALL_MAGNETIC_ELEMENTS
)
ANIONS = {'O', 'S', 'Se', 'Te', 'Po', 'F', 'Cl', 'Br', 'I', 'At'}

def parse_df_electrons(el):
    """
    Parses outer-shell d and f electron counts from electronic_structure string.
    Example: [Ar].3d6.4s2 -> d=6, f=0; [Xe].4f4.6s2 -> d=0, f=4.
    """
    d_count = 0.0
    f_count = 0.0
    estruc = el.electronic_structure
    if estruc:
        import re
        d_match = re.search(r'd(\d+)', estruc)
        if d_match:
            d_count = float(d_match.group(1))
        f_match = re.search(r'f(\d+)', estruc)
        if f_match:
            f_count = float(f_match.group(1))
    return d_count, f_count

def parse_outer_valence(el):
    """
    Robustly parses total outer valence-electron count from pymatgen's
    electronic_structure string. The previous digit-by-digit parser read d10 as
    1+0; this parser correctly treats d10 as 10.
    """
    estruc = el.electronic_structure
    if not estruc:
        return 0.0
    try:
        import re
        return float(sum(int(v) for v in re.findall(r'[spdf](\d+)', estruc)))
    except Exception:
        return 0.0

def is_rare_earth_elem(el):
    """Robust rare earth & actinide detector by atomic number (independent of pymatgen version naming)."""
    z = el.number
    return (57 <= z <= 71) or (89 <= z <= 103) or (el.symbol in {'Sc', 'Y'})

def get_element_node_features(el):
    """
    Extracts physical and chemical features for an element (22 Dimensions).
    """
    z = float(el.number)
    chi = float(el.X) if el.X is not None else 0.0
    group = float(el.group) if el.group is not None else 0.0
    period = float(el.row) if el.row is not None else 0.0
    
    # Outer electrons count as valence representation. Uses regex parser so d10/f10 are handled correctly.
    valence = parse_outer_valence(el)
    
    radius = float(el.atomic_radius) if el.atomic_radius is not None else 0.0
    
    # Upgraded structured magnetic priors (8 distinct physical groups)
    is_elemental_fm = 1.0 if el.symbol in ELEMENTAL_FM else 0.0
    is_elemental_afm = 1.0 if el.symbol in ELEMENTAL_AFM else 0.0
    is_strong_fm_3d = 1.0 if el.symbol in STRONG_FM_3D else 0.0
    is_strong_afm_3d = 1.0 if el.symbol in STRONG_AFM_3D else 0.0
    is_rare_earth_fm = 1.0 if el.symbol in RARE_EARTH_FM else 0.0
    is_rare_earth_afm = 1.0 if el.symbol in RARE_EARTH_AFM else 0.0
    is_complex_re_magnet = 1.0 if el.symbol in COMPLEX_RARE_EARTH_MAGNETS else 0.0
    is_molecular_afm = 1.0 if el.symbol in MOLECULAR_AFM_ELEMENTS else 0.0
    
    # --- UPGRADED NODE FEATURES ---
    d_count, f_count = parse_df_electrons(el)
    is_tm = 1.0 if el.is_transition_metal else 0.0
    is_re = 1.0 if is_rare_earth_elem(el) else 0.0
    is_anion = 1.0 if el.symbol in ANIONS else 0.0
    mendeleev_no = float(getattr(el, 'mendeleev_no', 0.0) or 0.0)
    rad = float(getattr(el, 'atomic_radius', 1.0) or 1.0)
    atomic_volume = float(4.0 / 3.0 * np.pi * (rad ** 3))
    ionization_energy = float(getattr(el, 'ionization_energy', 0.0) or 0.0)
        
    return [
        z, chi, group, period, valence, radius,
        is_elemental_fm, is_elemental_afm, is_strong_fm_3d, is_strong_afm_3d,
        is_rare_earth_fm, is_rare_earth_afm, is_complex_re_magnet, is_molecular_afm,
        d_count, f_count, is_tm, is_re, is_anion,
        mendeleev_no, atomic_volume, ionization_energy
    ]

def get_pair_edge_features(el_i, el_j, frac_i, frac_j):
    """
    Computes pairwise coupling/mismatch features between two elements (9 Dimensions).
    """
    chi_i = el_i.X if el_i.X is not None else 0.0
    chi_j = el_j.X if el_j.X is not None else 0.0
    d_chi = abs(chi_i - chi_j)
    
    r_i = el_i.atomic_radius if el_i.atomic_radius is not None else 0.0
    r_j = el_j.atomic_radius if el_j.atomic_radius is not None else 0.0
    d_r = abs(r_i - r_j)
    
    val_i = parse_outer_valence(el_i)
    val_j = parse_outer_valence(el_j)
    
    d_val = abs(val_i - val_j)
    f_ij = frac_i * frac_j
    
    # --- UPGRADED EDGE FEATURES ---
    is_mag_i = el_i.symbol in ALL_MAGNETIC_ELEMENTS
    is_mag_j = el_j.symbol in ALL_MAGNETIC_ELEMENTS
    mag_mag_flag = 1.0 if (is_mag_i and is_mag_j) else 0.0
    
    is_anion_i = el_i.symbol in ANIONS
    is_anion_j = el_j.symbol in ANIONS
    mag_anion_flag = 1.0 if ((is_mag_i and is_anion_j) or (is_mag_j and is_anion_i)) else 0.0
    
    is_re_i = is_rare_earth_elem(el_i)
    is_re_j = is_rare_earth_elem(el_j)
    is_tm_i = el_i.is_transition_metal
    is_tm_j = el_j.is_transition_metal
    re_tm_flag = 1.0 if ((is_re_i and is_tm_j) or (is_re_j and is_tm_i)) else 0.0
    
    d_i, f_i = parse_df_electrons(el_i)
    d_j, f_j = parse_df_electrons(el_j)
    d_diff = abs(d_i - d_j)
    f_diff = abs(f_i - f_j)
    
    return [d_chi, d_r, d_val, f_ij, mag_mag_flag, mag_anion_flag, re_tm_flag, d_diff, f_diff]

# ── Architecture Constants ────────────────────────────────────────────────────
NODE_DIM      = 128   # Hidden embedding dimension
EDGE_DIM      = 9     # Raw edge feature dimension (pair physics)
N_CONV_LAYERS = 3     # Number of Transformer blocks
MAX_ELEMENTS  = 10    # Max elements per formula; avoids truncating doped/high-entropy compositions

GRAPH_CACHE = {}
GRAPH_CACHE_FILE = "output/training/graph_cache.pkl"

def _load_graph_cache():
    global GRAPH_CACHE
    if os.path.exists(GRAPH_CACHE_FILE):
        try:
            with open(GRAPH_CACHE_FILE, 'rb') as f:
                GRAPH_CACHE.update(pickle.load(f))
        except Exception:
            pass

def _save_graph_cache():
    global GRAPH_CACHE
    try:
        os.makedirs(os.path.dirname(GRAPH_CACHE_FILE), exist_ok=True)
        with open(GRAPH_CACHE_FILE, 'wb') as f:
            pickle.dump(GRAPH_CACHE, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass

class CompositionGraphDataset(Dataset):
    """
    PyTorch Dataset that parses formula strings into element graph/transformer tokens.
    Supports Dense Padding for vectorized batched Self-Attention.
    """
    def __init__(self, formulas, targets=None, max_elements=MAX_ELEMENTS):
        self.formulas = list(formulas)
        self.max_elements = max_elements
        self.graphs = []
        
        if not GRAPH_CACHE:
            _load_graph_cache()
            
        if targets is not None:
            self.target_class = torch.tensor(targets.get('class', np.zeros(len(formulas))), dtype=torch.long)
            self.target_tc = torch.tensor(targets.get('tc', np.zeros(len(formulas))), dtype=torch.float32)
            self.target_tn = torch.tensor(targets.get('tn', np.zeros(len(formulas))), dtype=torch.float32)
        else:
            self.target_class = torch.zeros(len(formulas), dtype=torch.long)
            self.target_tc = torch.zeros(len(formulas), dtype=torch.float32)
            self.target_tn = torch.zeros(len(formulas), dtype=torch.float32)
            
        unseen_count = sum(1 for f in self.formulas if (str(f).strip() if f is not None else "") not in GRAPH_CACHE)
        if unseen_count > 0:
            print(f"Parsing {unseen_count} new chemical formulas into element embeddings (reusing {len(formulas) - unseen_count} cached)...")
            
        for formula in self.formulas:
            formula_str = str(formula).strip() if formula is not None else ""
            if formula_str in GRAPH_CACHE:
                self.graphs.append(GRAPH_CACHE[formula_str])
                continue
                
            try:
                comp = Composition(formula_str)
                elems = list(comp.elements)
                fracs = comp.element_composition.fractional_composition.as_dict()
                
                # Sort elements by Z to maintain node canonical order
                elems = sorted(elems, key=lambda e: e.number)
                n_nodes = len(elems)
                
                if n_nodes > self.max_elements:
                    elems = elems[:self.max_elements]
                    n_nodes = self.max_elements
                    
                node_feats = []
                for el in elems:
                    node_feats.append(get_element_node_features(el))
                node_feats = np.array(node_feats, dtype=np.float32)
                
                edge_feats = np.zeros((n_nodes, n_nodes, 9), dtype=np.float32)
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        el_i = elems[i]
                        el_j = elems[j]
                        frac_i = fracs.get(el_i.symbol, 0.0)
                        frac_j = fracs.get(el_j.symbol, 0.0)
                        edge_feats[i, j] = get_pair_edge_features(el_i, el_j, frac_i, frac_j)
                        
                node_fracs = np.array([fracs.get(el.symbol, 0.0) for el in elems], dtype=np.float32)
                
                graph_dict = {
                    'node_feats': node_feats,
                    'edge_feats': edge_feats,
                    'node_fracs': node_fracs,
                    'n_nodes': n_nodes
                }
                GRAPH_CACHE[formula_str] = graph_dict
                self.graphs.append(graph_dict)
            except Exception:
                # Fallback for parsing failure
                fallback = {
                    'node_feats': np.zeros((1, 22), dtype=np.float32),
                    'edge_feats': np.zeros((1, 1, 9), dtype=np.float32),
                    'node_fracs': np.array([1.0], dtype=np.float32),
                    'n_nodes': 1
                }
                GRAPH_CACHE[formula_str] = fallback
                self.graphs.append(fallback)
        _save_graph_cache()

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        t_class = self.target_class[idx]
        t_tc = self.target_tc[idx]
        t_tn = self.target_tn[idx]
        
        # Create padded tensors
        padded_nodes = np.zeros((self.max_elements, 22), dtype=np.float32)
        padded_edges = np.zeros((self.max_elements, self.max_elements, 9), dtype=np.float32)
        padded_fracs = np.zeros((self.max_elements,), dtype=np.float32)
        mask = np.zeros((self.max_elements,), dtype=np.float32)
        
        n = g['n_nodes']
        padded_nodes[:n] = g['node_feats']
        padded_edges[:n, :n] = g['edge_feats']
        padded_fracs[:n] = g['node_fracs']
        mask[:n] = 1.0
        
        return {
            'nodes': torch.tensor(padded_nodes),
            'edges': torch.tensor(padded_edges),
            'fracs': torch.tensor(padded_fracs),
            'mask': torch.tensor(mask),
            'target_class': t_class,
            'target_tc': t_tc,
            'target_tn': t_tn
        }

# ── Global Pre-trained State ───────────────────────────────────────────────────
_GLOBAL_PRETRAINED_STATE = None

# ═══════════════════════════════════════════════════════════════════════════════
# MAGFORMER: SELF-ATTENTION COMPOSITION TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════════════

class TransformerEncoderLayerWithBias(nn.Module):
    """
    Custom Transformer encoder layer supporting pairwise attention bias injection
    (e.g., electronegativity and ionic radius differences).
    """
    def __init__(self, d_model=128, n_heads=8, dim_feedforward=256, dropout=0.10):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, pair_bias, mask):
        # x: B x N x d_model
        # pair_bias: B x n_heads x N x N
        # mask: B x N
        B, N, D = x.shape
        
        # Pre-norm LayerNorm
        x_norm = self.norm1(x)
        
        # Project Queries, Keys, and Values
        q = self.q_proj(x_norm).view(B, N, self.n_heads, self.d_k).transpose(1, 2) # B x n_heads x N x d_k
        k = self.k_proj(x_norm).view(B, N, self.n_heads, self.d_k).transpose(1, 2) # B x n_heads x N x d_k
        v = self.v_proj(x_norm).view(B, N, self.n_heads, self.d_k).transpose(1, 2) # B x n_heads x N x d_k
        
        # Self-Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k) # B x n_heads x N x N
        
        # Add physics-based pairwise attention bias
        scores = scores + pair_bias
        
        # Key padding mask: mask out padding elements
        attn_mask = mask.unsqueeze(1).unsqueeze(2) # B x 1 x 1 x N
        scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Aggregated attention representations
        context = torch.matmul(attn_weights, v) # B x n_heads x N x d_k
        context = context.transpose(1, 2).contiguous().view(B, N, D)
        
        # Residual projection
        attn_out = self.out_proj(context)
        x = x + self.dropout(attn_out)
        
        # Feedforward MLP
        x = x + self.ffn(self.norm2(x))
        
        # Mask padded elements out of nodes
        x = x * mask.unsqueeze(-1)
        
        return x

class AttentionPooling(nn.Module):
    """
    Learnable attention-weighted pooling over valid elements.
    """
    def __init__(self, node_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.Tanh(),
            nn.Linear(node_dim // 2, 1)
        )
    
    def forward(self, nodes, mask, return_weights=False):
        attn_logits = self.attn(nodes).squeeze(-1) # B x N
        attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=-1) # B x N
        x_attn = torch.sum(nodes * attn_weights.unsqueeze(-1), dim=1) # B x D
        if return_weights:
            return x_attn, attn_weights
        return x_attn

class HeteroscedasticRegressionHead(nn.Module):
    """
    Predicts mean (mu) and log-variance (logvar) of log1p(T) transition temperature.
    Uses negative log-likelihood for heteroscedastic uncertainty modeling.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.15)
        )
        self.mean_head = nn.Linear(128, 1)
        self.logvar_head = nn.Linear(128, 1)
        
    def forward(self, x):
        feat = self.shared(x)
        mean = self.mean_head(feat).squeeze(-1)
        logvar = torch.clamp(self.logvar_head(feat).squeeze(-1), min=-10.0, max=5.0)
        return mean, logvar

class MagFormer(nn.Module):
    """
    MagFormer-MagNet: A Physics-Gated Formula Transformer for Magnetic Phase
    and Transition-Temperature Prediction with Gated Mixture-of-Experts and Tabular Stacking.
    """
    def __init__(self):
        super().__init__()
        
        # ── Input Embedding Layers ──────────────────────────────────────────────
        self.node_input_norm = nn.LayerNorm(22)
        # Feature-scale buffers keep the raw node matrix useful for element indexing
        # while presenting normalized physics descriptors to the neural network.
        self.register_buffer("node_scale", torch.tensor([
            118.0, 4.0, 18.0, 7.0, 32.0, 250.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            10.0, 14.0, 1.0, 1.0, 1.0,
            100.0, 100.0, 25.0
        ], dtype=torch.float32))
        self.register_buffer("edge_scale", torch.tensor([
            4.0, 250.0, 32.0, 1.0, 1.0, 1.0, 1.0, 10.0, 14.0
        ], dtype=torch.float32))
        self.element_embed = nn.Embedding(119, 64) # Index 0 to 118
        
        self.fraction_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 64)
        )
        
        self.physics_embed = nn.Sequential(
            nn.Linear(22, 32),
            nn.SiLU(),
            nn.Linear(32, 32)
        )
        
        # Project concatenated embeddings: element(64) + fraction(64) + physics(32) = 160D
        self.input_projection = nn.Linear(160, NODE_DIM)
        
        # ── Pair Attention Bias Network: 9D features -> n_heads (8) attention bias
        self.pair_bias_net = nn.Sequential(
            nn.Linear(9, 32),
            nn.SiLU(),
            nn.Linear(32, 8)
        )
        # Bounded and learned scaling factor to prevent self-attention saturation
        self.pair_bias_scale = nn.Parameter(torch.tensor(0.5))
        
        # Initialize pair_bias_net final layer to very small weights so bias starts small
        nn.init.normal_(self.pair_bias_net[-1].weight, std=0.01)
        nn.init.zeros_(self.pair_bias_net[-1].bias)
        
        # ── 3-Layer Transformer Blocks ──────────────────────────────────────────
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderLayerWithBias(d_model=NODE_DIM, n_heads=8, dim_feedforward=256, dropout=0.10)
            for _ in range(N_CONV_LAYERS)
        ])
        
        # ── Task-Specific Attention Pooling ──────────────────────────────────
        self.attn_pool_class = AttentionPooling(NODE_DIM)
        self.attn_pool_tc    = AttentionPooling(NODE_DIM)
        self.attn_pool_tn    = AttentionPooling(NODE_DIM)
        
        pooled_dim = 4 * NODE_DIM # frac + max + gmean + task_attn = 512
        
        # ── Shared Phase Classification Head ───────────────────────────────────
        self.class_head = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 3) # FM, AFM, NM logits
        )
        
        # ── Specialized Gated Expert Regression Heads ─────────────────────────
        self.tc_head = HeteroscedasticRegressionHead(pooled_dim) # FM expert
        self.tn_head = HeteroscedasticRegressionHead(pooled_dim) # AFM expert

    def forward(self, nodes, edges, fracs, mask):
        B, N, F_in = nodes.shape
        
        # Extract atomic number Z from raw first column for element embedding.
        z_indices = nodes[:, :, 0].long().clamp(0, 118)

        # Normalize physical node descriptors using fixed physically meaningful scales.
        nodes_scaled = nodes / self.node_scale.view(1, 1, -1)
        nodes_scaled = torch.nan_to_num(nodes_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        nodes_norm = self.node_input_norm(nodes_scaled.view(-1, F_in)).view(B, N, F_in)
        el_emb = self.element_embed(z_indices) # B x N x 64
        
        # Embed stoichiometry
        frac_emb = self.fraction_embed(fracs.unsqueeze(-1)) # B x N x 64
        
        # Embed physical descriptors
        phys_emb = self.physics_embed(nodes_norm) # B x N x 32
        
        # Concatenate and project embeddings
        x_concat = torch.cat([el_emb, frac_emb, phys_emb], dim=-1) # B x N x 160
        x = self.input_projection(x_concat) * mask.unsqueeze(-1) # B x N x 128
        
        # Compute pairwise attention bias from scaled edge descriptors.
        edges_scaled = edges / self.edge_scale.view(1, 1, 1, -1)
        edges_scaled = torch.nan_to_num(edges_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        pair_bias = self.pair_bias_net(edges_scaled) # B x N x N x 8
        pair_bias = torch.tanh(pair_bias) * self.pair_bias_scale # Bounded and scaled
        pair_bias = pair_bias.permute(0, 3, 1, 2) # B x 8 x N x N
        
        # Pass tokens through the Composition Transformer Blocks
        for layer in self.transformer_blocks:
            x = layer(x, pair_bias, mask)
            
        # ── 4-Way Element-Wise Aggregation ──────────────────────────────────
        # 1. Stoichiometry-aware fractional weighted mean
        fracs_weight = fracs.unsqueeze(-1)
        x_frac = torch.sum(x * fracs_weight, dim=1) # B x 128
        
        # 2. Masked max pooling (strongest elemental signal)
        x_masked = x.clone()
        x_masked[mask == 0] = -1e9
        x_max, _ = torch.max(x_masked, dim=1) # B x 128
        
        # 3. Masked global mean pooling (unweighted robust average)
        n_valid_flat = mask.sum(dim=1, keepdim=True).clamp(min=1)
        x_gmean = (x * mask.unsqueeze(-1)).sum(dim=1) / n_valid_flat # B x 128
        
        # 4. Task-specific attention pooling
        x_attn_class = self.attn_pool_class(x, mask)
        x_attn_tc    = self.attn_pool_tc(x, mask)
        x_attn_tn    = self.attn_pool_tn(x, mask)
        
        # Build pooled representation vectors (512D)
        x_pooled_class = torch.cat([x_frac, x_max, x_gmean, x_attn_class], dim=-1)
        x_pooled_tc    = torch.cat([x_frac, x_max, x_gmean, x_attn_tc], dim=-1)
        x_pooled_tn    = torch.cat([x_frac, x_max, x_gmean, x_attn_tn], dim=-1)
        
        # Compute task predictions
        logits_class = self.class_head(x_pooled_class)
        pred_tc_mean, pred_tc_logvar = self.tc_head(x_pooled_tc)
        pred_tn_mean, pred_tn_logvar = self.tn_head(x_pooled_tn)
        
        return logits_class, (pred_tc_mean, pred_tc_logvar), (pred_tn_mean, pred_tn_logvar)

    def extract_embeddings(self, nodes, edges, fracs, mask):
        """
        Extracts the shared 384-dimensional pooled composition representation
        (stoichiometry-weighted mean + masked max + global mean).
        Excludes task-specific attention pooling to prevent target leakage.
        """
        B, N, F_in = nodes.shape
        z_indices = nodes[:, :, 0].long().clamp(0, 118)
        nodes_scaled = nodes / self.node_scale.view(1, 1, -1)
        nodes_scaled = torch.nan_to_num(nodes_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        nodes_norm = self.node_input_norm(nodes_scaled.view(-1, F_in)).view(B, N, F_in)
        el_emb = self.element_embed(z_indices)
        frac_emb = self.fraction_embed(fracs.unsqueeze(-1))
        phys_emb = self.physics_embed(nodes_norm)
        x_concat = torch.cat([el_emb, frac_emb, phys_emb], dim=-1)
        x = self.input_projection(x_concat) * mask.unsqueeze(-1)
        edges_scaled = edges / self.edge_scale.view(1, 1, 1, -1)
        edges_scaled = torch.nan_to_num(edges_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        pair_bias = self.pair_bias_net(edges_scaled)
        pair_bias = torch.tanh(pair_bias) * self.pair_bias_scale
        pair_bias = pair_bias.permute(0, 3, 1, 2)
        for layer in self.transformer_blocks:
            x = layer(x, pair_bias, mask)
        # Shared pooling (no task-specific attention to avoid target leakage)
        fracs_weight = fracs.unsqueeze(-1)
        x_frac = torch.sum(x * fracs_weight, dim=1)
        x_masked = x.clone()
        x_masked[mask == 0] = -1e9
        x_max, _ = torch.max(x_masked, dim=1)
        n_valid_flat = mask.sum(dim=1, keepdim=True).clamp(min=1)
        x_gmean = (x * mask.unsqueeze(-1)).sum(dim=1) / n_valid_flat
        return torch.cat([x_frac, x_max, x_gmean], dim=-1)  # (B, 384)

# Backward-compatible alias
CompositionGNN = MagFormer

# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS: FOCAL LOSS & HETEROSCEDASTIC REGRESSION LOSS
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss with minority class weighting and asymmetric confusion cost.
    FL = -alpha * cost * (1 - pt)^gamma * log(pt)
    
    The confusion_cost matrix applies extra penalty for specific misclassification
    pairs (e.g., AFM↔NM confusion gets 2× penalty).
    """
    def __init__(self, alpha=None, gamma=2.0, confusion_cost=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.confusion_cost = confusion_cost  # (n_classes, n_classes) tensor
        self.reduction = reduction
        
    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # Apply asymmetric confusion cost: penalize specific misclassification pairs
        if self.confusion_cost is not None:
            probs = torch.softmax(logits.detach(), dim=-1)  # (B, C)
            # For each sample, compute expected confusion cost: sum_j P(j) * cost[true, j]
            cost_per_sample = (probs * self.confusion_cost[targets]).sum(dim=-1)  # (B,)
            focal_loss = focal_loss * cost_per_sample
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def heteroscedastic_nll_loss(mean, logvar, target):
    """
    Gaussian negative log likelihood loss.
    Loss = 0.5 * exp(-logvar) * (target - mean)^2 + 0.5 * logvar
    """
    precision = torch.exp(-logvar)
    squared_error = (target - mean) ** 2
    loss = 0.5 * precision * squared_error + 0.5 * logvar
    return loss.mean()

# ═══════════════════════════════════════════════════════════════════════════════
# MAGFORMER TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_composition_gnn_model(formulas, targets, epochs=30, batch_size=512, verbose=True):
    """
    Trains the MagFormer model with Focal Loss and Gated Heteroscedastic Experts.
    Uses OneCycleLR and AdamW optimization. Warm-starts from _GLOBAL_PRETRAINED_STATE if available.
    """
    dataset = CompositionGraphDataset(formulas, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_steps_per_epoch = len(dataloader)

    
    model = MagFormer()
    
    # Warm-start logic
    if _GLOBAL_PRETRAINED_STATE is not None:
        try:
            model.load_state_dict(_GLOBAL_PRETRAINED_STATE, strict=False)
            if verbose:
                print(f"  [MagFormer] Warm-starting from global pre-trained state. Fine-tuning for {epochs} epochs...")
        except Exception as e:
            if verbose:
                print(f"  [MagFormer] Warm-start loading failed ({e}). Training from scratch...")
                
    # Class weights for Focal Loss
    targets_class = np.asarray(targets.get('class', np.zeros(len(formulas))), dtype=int)
    class_counts = np.bincount(targets_class, minlength=3)
    class_counts = np.maximum(class_counts, 1)
    class_weights = len(targets_class) / (3.0 * class_counts.astype(float))
    class_weights[1] *= 1.4 # Moderate AFM upweighting (was 2.0 — overcorrected, causing low AFM precision)
    alpha_tensor = torch.tensor(class_weights, dtype=torch.float32)
    
    criterion_class = FocalLoss(alpha=alpha_tensor, gamma=2.0, confusion_cost=torch.tensor([
        #  FM   AFM   NM     ← predicted class
        [1.0, 1.0, 1.0],   # true = FM: normal cost for all errors
        [1.0, 1.0, 2.0],   # true = AFM: 2× penalty when predicted NM
        [1.0, 2.0, 1.0],   # true = NM:  2× penalty when predicted AFM
    ], dtype=torch.float32))
    huber_loss_fn = nn.HuberLoss(delta=1.0)
    
    # Optimizers — tuned for balanced classification/regression loss with confusion-aware FocalLoss
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, epochs=epochs,
        steps_per_epoch=n_steps_per_epoch,
        pct_start=0.20, div_factor=5.0, final_div_factor=100.0
    )
    
    # GNN is used mainly as an auxiliary composition encoder for stacking.
    # Classification receives the strongest weight; TC/TN act as soft physical priors.
    w_class, w_tc, w_tn = 3.0, 1.0, 1.0
    
    model.train()
    if verbose:
        print(f"  Training MagFormer (NODE_DIM=128, FocalLoss, Heteroscedastic Gated Experts) for {epochs} epochs...")
    history = {
        'epoch': [],
        'loss_total': [],
        'loss_class': [],
        'loss_tc': [],
        'loss_tn': [],
        'lr': []
    }

    for epoch in range(epochs):
        epoch_loss = epoch_loss_class = epoch_loss_tc = epoch_loss_tn = 0.0
        
        for batch in dataloader:
            nodes  = batch['nodes']
            edges  = batch['edges']
            fracs  = batch['fracs']
            mask   = batch['mask']
            y_class = batch['target_class']
            y_tc    = batch['target_tc']
            y_tn    = batch['target_tn']
            
            optimizer.zero_grad()
            
            logits_class, (pred_tc, tc_logvar), (pred_tn, tn_logvar) = model(nodes, edges, fracs, mask)
            
            # 1. Classification Focal Loss
            loss_class = criterion_class(logits_class, y_class)
            
            # 2. TC Regression Loss - Heteroscedastic NLL + Huber Loss for ALL samples with valid TC
            tc_mask = (~torch.isnan(y_tc)) & (y_tc > 0) & (y_class == 0)
            if tc_mask.any():
                y_tc_log = torch.log1p(y_tc[tc_mask])
                nll_loss_tc = heteroscedastic_nll_loss(pred_tc[tc_mask], tc_logvar[tc_mask], y_tc_log)
                aux_loss_tc = huber_loss_fn(pred_tc[tc_mask], y_tc_log)
                loss_tc = nll_loss_tc + 1.0 * aux_loss_tc
            else:
                loss_tc = y_tc.new_tensor(0.0)
                
            # 3. TN Regression Loss - Heteroscedastic NLL + Huber Loss for ALL samples with valid TN
            tn_mask = (~torch.isnan(y_tn)) & (y_tn > 0) & (y_class == 1)
            if tn_mask.any():
                y_tn_log = torch.log1p(y_tn[tn_mask])
                nll_loss_tn = heteroscedastic_nll_loss(pred_tn[tn_mask], tn_logvar[tn_mask], y_tn_log)
                aux_loss_tn = huber_loss_fn(pred_tn[tn_mask], y_tn_log)
                loss_tn = nll_loss_tn + 1.0 * aux_loss_tn
            else:
                loss_tn = y_tn.new_tensor(0.0)
                
            loss = w_class * loss_class + w_tc * loss_tc + w_tn * loss_tn
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            
            n = len(nodes)
            epoch_loss       += loss.item() * n
            epoch_loss_class += loss_class.item() * n
            epoch_loss_tc    += loss_tc.item() * n
            epoch_loss_tn    += loss_tn.item() * n
            
        N = len(dataset)
        epoch_loss       /= N
        epoch_loss_class /= N
        epoch_loss_tc    /= N
        epoch_loss_tn    /= N
        lr = optimizer.param_groups[0]['lr']
        
        history['epoch'].append(epoch + 1)
        history['loss_total'].append(float(epoch_loss))
        history['loss_class'].append(float(epoch_loss_class))
        history['loss_tc'].append(float(epoch_loss_tc))
        history['loss_tn'].append(float(epoch_loss_tn))
        history['lr'].append(float(lr))
        
        if verbose:
            print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Loss = {epoch_loss:.5f} "
                  f"| Class = {epoch_loss_class:.4f} | TC = {epoch_loss_tc:.4f} "
                  f"| TN = {epoch_loss_tn:.4f} | LR = {lr:.6f}")
                  
    model.training_history = history
    return model

def predict_with_gnn(model, formulas, task_type='classification', batch_size=512):
    """
    Generates predictions from the MagFormer composition transformer.
    Returns:
    - 'classification': soft probabilities (N, 3)
    - 'curie' / 'tc':   Curie temperatures in Kelvin (N,) [stable expm1(mu) median back-transform]
    - 'neel' / 'tn':    Néel temperatures in Kelvin (N,) [stable expm1(mu) median back-transform]
    - 'all':            tuple (class_probs, pred_tc, pred_tn)
    - 'all_with_uncertainty': tuple (class_probs, pred_tc, pred_tn, tc_std, tn_std)
    """
    dataset = CompositionGraphDataset(formulas, targets=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_class_probs = []
    all_pred_tc, all_pred_tn = [], []
    all_std_tc, all_std_tn = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            nodes = batch['nodes']
            edges = batch['edges']
            fracs = batch['fracs']
            mask  = batch['mask']
            
            logits_class, (pred_tc, tc_logvar), (pred_tn, tn_logvar) = model(nodes, edges, fracs, mask)
            
            # Soft probabilities
            probs = torch.softmax(logits_class, dim=-1).numpy()
            all_class_probs.append(probs)
            
            # Back-transform from log1p space using exact Log-Normal expectations & variances
            mu_tc = np.clip(pred_tc.numpy(), -2.0, 8.0)
            var_tc = np.clip(torch.exp(tc_logvar).numpy(), 1e-6, 4.0)
            
            # Use the log-normal median as the auxiliary temperature feature.
            # This is more stable than the expectation exp(mu + 0.5*var)-1 when uncertainty is large.
            tc_expected = np.expm1(mu_tc)
            tc_expected = np.clip(tc_expected, 0.0, 2500.0) # conservative physical clamp
            all_pred_tc.append(tc_expected)
            
            # Standard Deviation: Var[T] = (exp(var) - 1) * exp(2*mu + var)
            tc_var_T = (np.exp(var_tc) - 1.0) * np.exp(2.0 * mu_tc + var_tc)
            tc_std_T = np.sqrt(np.maximum(tc_var_T, 0.0))
            tc_std_T = np.clip(tc_std_T, 0.0, 1000.0)
            all_std_tc.append(tc_std_T)
            
            # Same for Neel
            mu_tn = np.clip(pred_tn.numpy(), -2.0, 8.0)
            var_tn = np.clip(torch.exp(tn_logvar).numpy(), 1e-6, 4.0)
            
            tn_expected = np.expm1(mu_tn)
            tn_expected = np.clip(tn_expected, 0.0, 2500.0)
            all_pred_tn.append(tn_expected)
            
            tn_var_T = (np.exp(var_tn) - 1.0) * np.exp(2.0 * mu_tn + var_tn)
            tn_std_T = np.sqrt(np.maximum(tn_var_T, 0.0))
            tn_std_T = np.clip(tn_std_T, 0.0, 1000.0)
            all_std_tn.append(tn_std_T)
            
    class_probs = np.concatenate(all_class_probs, axis=0)
    pred_tc     = np.concatenate(all_pred_tc, axis=0)
    pred_tn     = np.concatenate(all_pred_tn, axis=0)
    std_tc      = np.concatenate(all_std_tc, axis=0)
    std_tn      = np.concatenate(all_std_tn, axis=0)
    
    if task_type == 'classification':
        return class_probs
    elif task_type in ['regression', 'curie', 'tc']:
        return pred_tc
    elif task_type in ['neel', 'tn']:
        return pred_tn
    elif task_type == 'all':
        return class_probs, pred_tc, pred_tn
    elif task_type == 'all_with_uncertainty':
        return class_probs, pred_tc, pred_tn, std_tc, std_tn
    else:
        return class_probs, pred_tc, pred_tn

def extract_magformer_embeddings(model, formulas, batch_size=512):
    """
    Extracts learned 384-dimensional composition embeddings from a trained MagFormer.
    These embeddings encode non-linear element interactions learned via self-attention
    and can be injected as features into tabular models (XGBoost, LightGBM, etc.).

    Returns: numpy array of shape (N, 384).
    """
    dataset = CompositionGraphDataset(formulas, targets=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            nodes = batch['nodes']
            edges = batch['edges']
            fracs = batch['fracs']
            mask = batch['mask']
            emb = model.extract_embeddings(nodes, edges, fracs, mask)
            all_embeddings.append(emb.numpy())

    return np.concatenate(all_embeddings, axis=0)

# ═══════════════════════════════════════════════════════════════════════════════
# PRE-TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def set_global_pretrained_state(state_dict):
    """
    Sets the pre-trained warm-start state_dict for fold fine-tuning.
    """
    global _GLOBAL_PRETRAINED_STATE
    _GLOBAL_PRETRAINED_STATE = state_dict

def pretrain_gnn_encoder(all_formulas, all_targets, epochs=40, batch_size=512, verbose=True):
    """
    Pre-trains MagFormer on all available formulas in one single pass.
    Warm-starts all downstream cross-validation fold training blocks.
    """
    global _GLOBAL_PRETRAINED_STATE
    
    if _GLOBAL_PRETRAINED_STATE is not None:
        if verbose:
            print("  [MagFormer Pre-training] Reusing cached global pre-trained state.")
        return _GLOBAL_PRETRAINED_STATE
        
    if verbose:
        print(f"\n  [MagFormer Pre-training] Training on {len(all_formulas)} materials for {epochs} epochs...")
        print("  (Run once. All CV fold models will warm-start fine-tuning from these weights.)")
        
    saved_state = _GLOBAL_PRETRAINED_STATE
    _GLOBAL_PRETRAINED_STATE = None
    
    pretrained_model = train_composition_gnn_model(
        all_formulas, all_targets,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )
    
    _GLOBAL_PRETRAINED_STATE = pretrained_model.state_dict()
    
    if verbose:
        print("  [MagFormer Pre-training] Complete. Global state set.")
        
    return _GLOBAL_PRETRAINED_STATE


def save_magformer_model(model, filepath="output/training/magformer_model.pt"):
    """Saves MagFormer model state dict and architecture metadata."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    payload = {
        'state_dict': model.state_dict(),
        'node_dim': NODE_DIM,
        'edge_dim': EDGE_DIM,
        'n_conv_layers': N_CONV_LAYERS,
        'max_elements': MAX_ELEMENTS,
    }
    torch.save(payload, filepath)
    print(f"Saved MagFormer model checkpoint to {filepath}")


def load_magformer_model(filepath="output/training/magformer_model.pt"):
    """Loads a pre-trained MagFormer model."""
    if not os.path.exists(filepath):
        return None
    try:
        payload = torch.load(filepath, map_location='cpu')
        model = MagFormer()
        if isinstance(payload, dict) and 'state_dict' in payload:
            model.load_state_dict(payload['state_dict'])
        elif isinstance(payload, dict):
            model.load_state_dict(payload)
        elif isinstance(payload, nn.Module):
            model = payload
        model.eval()
        return model
    except Exception as e:
        print(f"Warning: Could not load MagFormer checkpoint from {filepath}: {e}")
        return None


def extract_magformer_attention_weights(model, formula: str) -> Dict[str, float]:
    """
    Extracts element-level attention weights from MagFormer for an input formula.
    Provides physical interpretability on which elements dominate the magnetic interaction.
    """
    try:
        from pymatgen.core import Composition
        comp = Composition(str(formula).strip())
        el_list = [el.symbol for el in comp.elements]
        if not el_list:
            return {}
        if model is None:
            tot = sum(comp.values())
            return {el: float(comp[el] / tot) for el in el_list}
            
        dataset = CompositionGraphDataset([formula], targets=None)
        if len(dataset) == 0:
            return {}
        batch = dataset[0]
        nodes = batch['nodes'].unsqueeze(0)
        edges = batch['edges'].unsqueeze(0)
        fracs = batch['fracs'].unsqueeze(0)
        mask = batch['mask'].unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            el_idx = nodes[:, :, 0].long().clamp(0, 118)
            el_emb = model.element_embed(el_idx)
            frac_emb = model.fraction_embed(fracs.unsqueeze(-1))
            nodes_norm = model.node_input_norm(nodes / model.node_scale.view(1, 1, -1))
            phys_emb = model.physics_embed(nodes_norm)
            x_concat = torch.cat([el_emb, frac_emb, phys_emb], dim=-1)
            x = model.input_projection(x_concat) * mask.unsqueeze(-1)
            
            edges_scaled = edges / model.edge_scale.view(1, 1, 1, -1)
            edges_scaled = torch.nan_to_num(edges_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            pair_bias = model.pair_bias_net(edges_scaled)
            pair_bias = torch.tanh(pair_bias) * model.pair_bias_scale
            pair_bias = pair_bias.permute(0, 3, 1, 2)
            
            for layer in model.transformer_blocks:
                x = layer(x, pair_bias, mask)
                
            _, attn_weights = model.attn_pool_class(x, mask, return_weights=True)
            weights = attn_weights[0].numpy()
            
            out = {}
            for idx, el in enumerate(el_list):
                if idx < len(weights):
                    out[el] = float(weights[idx])
            s = sum(out.values())
            if s > 0:
                out = {k: round(v / s, 4) for k, v in out.items()}
            return out
    except Exception:
        return {}
