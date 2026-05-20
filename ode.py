"""
Neural ODE-RNN for Irregular Temporal Data — AKD Prediction

Architecture (ODE-RNN, Chen et al. 2018 / Rubanova et al. 2019):
  Between observations:
      dh/dt = ODEFunc(h, t)     — neural network defines hidden state dynamics
      h(t_next) = ODESolve(ODEFunc, h(t_prev), t_prev, t_next)   [RK4]
  At each observation:
      h_new = GRUCell([x_t, m_t], h(t))   — assimilate new data

  ODEModel (end-to-end nn.Module):
      ODE evolution + GRU update at each step → MLP classifier → sigmoid

Key advantage over discrete RNNs:
  The ODE naturally handles irregular gaps — larger gaps produce more hidden-
  state evolution before the GRU update, smaller gaps produce less.

References:
  Chen et al. (2018) "Neural Ordinary Differential Equations", NeurIPS.
  Rubanova et al. (2019) "Latent ODEs for Irregularly-Sampled Time Series", NeurIPS.
"""

import os
import sys
import copy
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PT)

from utils.prepare_data import trainTestPatients
from TimeEmbedding import (
    DEVICE,
    get_all_temporal_features,
    IrregularTimeSeriesDataset,
    collate_fn,
    load_and_prepare_patients,
)
from TimeEmbeddingVal import split_patients_train_val


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
seed_everything(SEED)


# ---------------------------------------------------------------------------
# ODE function: dh/dt = f(h, t)
# ---------------------------------------------------------------------------

class ODEFunc(nn.Module):
    """
    Neural network defining the ODE dynamics: dh/dt = f(h, t).

    A two-layer MLP with tanh activations.  Tanh bounds the derivative,
    which stabilises integration and prevents hidden-state explosion.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # Track NFE for diagnostics (no grad)
        self.nfe = 0

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        h: [B, H]  current hidden state
        t: [B]     current time (not used here — autonomous ODE, kept for API)
        Returns dh/dt: [B, H]
        """
        self.nfe += 1
        return self.net(h)


# ---------------------------------------------------------------------------
# RK4 solver with per-sample variable step sizes
# ---------------------------------------------------------------------------

def rk4_solve(func: ODEFunc, h: torch.Tensor, dt: torch.Tensor,
              n_steps: int = 4) -> torch.Tensor:
    """
    Integrate h from 0 to dt using classical 4th-order Runge-Kutta.

    Using n_steps fixed sub-steps guarantees bounded error even for
    large time gaps (up to ~24 h in this dataset).

    Args:
        func:    ODEFunc  — dh/dt = func(h, t)
        h:       [B, H]  — initial hidden state
        dt:      [B]     — per-sample integration intervals (hours)
        n_steps: number of equal sub-steps

    Returns:
        h:  [B, H]  — state after integration
    """
    if n_steps < 1:
        n_steps = 1

    step = dt / n_steps              # [B]
    step_col = step.unsqueeze(-1)    # [B, 1] — for broadcasting with [B, H]
    t = torch.zeros_like(dt)

    for _ in range(n_steps):
        k1 = func(h,                      t)
        k2 = func(h + 0.5 * step_col * k1, t + 0.5 * step)
        k3 = func(h + 0.5 * step_col * k2, t + 0.5 * step)
        k4 = func(h +       step_col * k3, t +       step)
        h  = h + (step_col / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t  = t + step

    return h


# ---------------------------------------------------------------------------
# End-to-end ODE-RNN model
# ---------------------------------------------------------------------------

class ODEModel(nn.Module):
    """
    End-to-end ODE-RNN for irregular time-series classification.

    For each step t in the sequence:
      1. Evolve hidden state from t-1 → t using the Neural ODE (RK4).
      2. Assimilate the observation via a GRU cell: h ← GRU([x, m], h).
      3. Mask updates for padding positions.

    Final hidden state → MLP classifier → P(AKD).
    """

    def __init__(self, input_dim: int, hidden_dim: int, ode_steps: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ode_steps  = ode_steps

        # Continuous dynamics
        self.ode_func = ODEFunc(hidden_dim)

        # GRU cell: input = [x (D), mask (D)]
        self.gru_cell = nn.GRUCell(input_size=input_dim * 2, hidden_size=hidden_dim)

        # Learned initial hidden state
        self.h0 = nn.Parameter(torch.zeros(hidden_dim))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch_data: dict) -> torch.Tensor:
        """
        Args:
            batch_data: dict with keys
                times   [B, T]    — observation times in hours
                values  [B, T, D] — feature values (zero where unobserved)
                masks   [B, T, D] — binary observation mask
                lengths [B]       — true sequence lengths
        Returns:
            probs [B] — P(AKD positive)
        """
        device = self.h0.device
        times   = batch_data["times"].to(device)    # [B, T]
        values  = batch_data["values"].to(device)   # [B, T, D]
        masks   = batch_data["masks"].to(device)    # [B, T, D]
        lengths = batch_data["lengths"]             # [B]

        B, T = times.shape

        h = self.h0.unsqueeze(0).expand(B, -1).contiguous()  # [B, H]

        for t in range(T):
            # ---- Step 1: ODE evolution ----
            if t > 0:
                # Per-sample time delta (hours); clamp to [0, ∞) for safety
                dt = (times[:, t] - times[:, t - 1]).clamp(min=0.0)  # [B]
                h = rk4_solve(self.ode_func, h, dt, n_steps=self.ode_steps)

            # ---- Step 2: GRU update at observation ----
            gru_in = torch.cat([values[:, t], masks[:, t]], dim=-1)  # [B, 2D]
            h_new  = self.gru_cell(gru_in, h)

            # ---- Step 3: mask out padding positions ----
            active = (t < lengths).to(device).float().unsqueeze(-1)
            h = active * h_new + (1.0 - active) * h

        return self.classifier(h).squeeze(-1)  # [B]


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_model(model: ODEModel, train_loader: DataLoader,
                val_loader: DataLoader, num_epochs: int = 100,
                eval_every: int = 5, patience: int = 10,
                lr: float = 5e-4) -> ODEModel:
    """Train with validation monitoring and early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_auc = 0.0
    best_state   = None
    no_improve   = 0

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        model.ode_func.nfe = 0
        total_loss = 0.0

        for batch_data, labels in train_loader:
            labels = labels.to(DEVICE)
            preds  = model(batch_data)
            loss   = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # ---- Validation ----
        if (epoch + 1) % eval_every == 0:
            model.eval()
            val_probs, val_labels = [], []
            with torch.no_grad():
                for batch_data, labels in val_loader:
                    val_probs.extend(model(batch_data).cpu().numpy())
                    val_labels.extend(labels.numpy())

            val_auc  = roc_auc_score(val_labels, val_probs)
            avg_loss = total_loss / len(train_loader)
            nfe      = model.ode_func.nfe

            print(f"    Epoch {epoch+1:3d} | loss {avg_loss:.4f} | "
                  f"val AUC {val_auc:.4f} | NFE {nfe}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state   = copy.deepcopy(model.state_dict())
                no_improve   = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    Early stopping at epoch {epoch + 1}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"    Best val AUC: {best_val_auc:.4f}")
    return model


def evaluate_model(model: ODEModel, loader: DataLoader):
    """Return (labels, probs, binary_preds) arrays."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch_data, labels in loader:
            all_probs.extend(model(batch_data).cpu().numpy())
            all_labels.extend(labels.numpy())
    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs > 0.5).astype(int)
    return labels, probs, preds


# ---------------------------------------------------------------------------
# Main — cross-validation
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Neural ODE-RNN for AKD Prediction (Irregular Temporal Data)")
    print("=" * 80)

    patients       = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)
    input_dim      = len(temporal_feats)
    print(f"Temporal features: {input_dim}")

    metrics = {k: [] for k in ["auc", "auc_pr", "acc", "rec", "spec", "prec"]}

    fig, ax = plt.subplots(figsize=(9, 7))

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, seed=SEED)):
        print(f"\n{'='*80}")
        print(f"Fold {fold}")
        print("=" * 80)

        train_p_obj, val_p_obj = split_patients_train_val(
            train_full, val_ratio=0.1, seed=SEED + fold
        )

        # Datasets (reuse IrregularTimeSeriesDataset from TimeEmbedding)
        train_ds = IrregularTimeSeriesDataset(train_p_obj, temporal_feats)
        stats    = train_ds.get_normalization_stats()
        val_ds   = IrregularTimeSeriesDataset(val_p_obj, temporal_feats, stats)
        test_ds  = IrregularTimeSeriesDataset(test_p,    temporal_feats, stats)

        print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  collate_fn=collate_fn)
        val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, collate_fn=collate_fn)
        test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, collate_fn=collate_fn)

        # Model
        model = ODEModel(
            input_dim=input_dim,
            hidden_dim=64,
            ode_steps=4,
        ).to(DEVICE)

        print("  Training ODE-RNN...")
        model = train_model(
            model, train_loader, val_loader,
            num_epochs=120, eval_every=5, patience=8, lr=5e-4,
        )

        # Evaluation
        y_true, y_prob, y_pred = evaluate_model(model, test_loader)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)

        fold_auc  = roc_auc_score(y_true, y_prob)
        fold_aupr = auc(rec_arr, prec_arr)
        fold_acc  = accuracy_score(y_true, y_pred)
        fold_rec  = recall_score(y_true, y_pred, zero_division=0)
        fold_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fold_prec = precision_score(y_true, y_pred, zero_division=0)

        metrics["auc"].append(fold_auc)
        metrics["auc_pr"].append(fold_aupr)
        metrics["acc"].append(fold_acc)
        metrics["rec"].append(fold_rec)
        metrics["spec"].append(fold_spec)
        metrics["prec"].append(fold_prec)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC={fold_auc:.3f})")

        print(f"  Fold {fold} → AUC {fold_auc:.4f} | AUPR {fold_aupr:.4f} | "
              f"Sens {fold_rec:.4f} | Spec {fold_spec:.4f}")

    # ROC plot
    ax.plot([0, 1], [0, 1], "--", color="navy", lw=2, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Neural ODE-RNN: AKD Prediction (Irregular Temporal)", fontsize=14,
                 fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs("result", exist_ok=True)
    plt.savefig("result/ode.png", dpi=300)
    print("\nROC plot saved to result/ode.png")

    # Summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY — Neural ODE-RNN")
    print("=" * 80)

    def stat(name, vals):
        print(f"{name:22s} | {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    stat("AUC",                metrics["auc"])
    stat("AUC-PR",             metrics["auc_pr"])
    stat("Accuracy",           metrics["acc"])
    stat("Sensitivity/Recall", metrics["rec"])
    stat("Specificity",        metrics["spec"])
    stat("Precision",          metrics["prec"])
    print("=" * 80)


if __name__ == "__main__":
    main()
