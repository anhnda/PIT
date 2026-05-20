"""
GRU-D: Gated Recurrent Units with Decay for Irregular Temporal Data

Architecture:
  GRUDCell:
    - Input decay:  γ_x^d = exp(-max(0, W_γx * δ^d + b)) per feature
                    x̂ = m * x + (1-m) * (γ_x * x_last + (1-γ_x) * x̄)
    - Hidden decay: γ_h = exp(-max(0, W_γh * δ + b))
                    h̃ = γ_h ⊙ h
    - GRU update on [x̂, m] with decayed hidden state h̃

  GRUDModel (end-to-end nn.Module):
    GRUDCell → MLP classifier → sigmoid

Reference: Che et al. (2018) "Recurrent Neural Networks for Multivariate Time
Series with Missing Values", Scientific Reports.
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
from torch.utils.data import Dataset, DataLoader

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
# Path setup (RIT project directory)
# ---------------------------------------------------------------------------
PT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PT)

from constants import NULLABLE_MEASURES
from utils.class_patient import Patients
from utils.prepare_data import trainTestPatients

from TimeEmbedding import (
    FIXED_FEATURES,
    DEVICE,
    get_all_temporal_features,
    extract_temporal_data,
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
# GRU-D specific data helpers
# ---------------------------------------------------------------------------

def compute_grud_features(times_hours, values, masks):
    """
    Pre-compute per-feature time gaps (δ) and carry-forward last values for GRU-D.

    δ_t^d = time elapsed since the last observation of feature d before step t.
    x_last_t^d = last observed value of feature d at or before step t.

    Returns:
        deltas: ndarray [T, D]  — hours since last observation (0 if never seen)
        x_lasts: ndarray [T, D] — carry-forward of last observed value
    """
    T = len(times_hours)
    D = len(values[0])

    deltas = np.zeros((T, D), dtype=np.float32)
    x_lasts = np.zeros((T, D), dtype=np.float32)

    last_obs_time = [None] * D   # time of the last observation per feature
    last_obs_val = [0.0] * D     # value at the last observation per feature

    for t in range(T):
        t_now = times_hours[t]
        for d in range(D):
            if masks[t][d] > 0:
                # Feature d is observed at this step
                if last_obs_time[d] is None:
                    deltas[t][d] = 0.0
                else:
                    deltas[t][d] = t_now - last_obs_time[d]
                last_obs_time[d] = t_now
                last_obs_val[d] = values[t][d]
            else:
                # Feature d is not observed — carry forward gap
                if last_obs_time[d] is None:
                    deltas[t][d] = 0.0
                else:
                    deltas[t][d] = t_now - last_obs_time[d]
            x_lasts[t][d] = last_obs_val[d]

    return deltas, x_lasts


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GRUDDataset(Dataset):
    """
    Dataset that precomputes all GRU-D inputs:
      values, masks, deltas (δ), x_lasts (carry-forward), and labels.
    """

    def __init__(self, patients, feature_names, normalization_stats=None):
        self.feature_names = feature_names
        self.data = []
        self.labels = []

        patient_list = patients.patientList if hasattr(patients, "patientList") else patients

        all_observed = []  # collect all observed normalized values for x_mean

        # ---- Pass 1: collect raw temporal data ----
        raw_records = []
        for patient in patient_list:
            times, values, masks = extract_temporal_data(patient, feature_names)
            if times is None:
                continue
            raw_records.append((times, values, masks, 1 if patient.akdPositive else 0))
            for v_vec, m_vec in zip(values, masks):
                for v, m in zip(v_vec, m_vec):
                    if m > 0:
                        all_observed.append(v)

        # ---- Normalization stats (global scalar, consistent with existing code) ----
        if normalization_stats is None:
            arr = np.array(all_observed)
            self.mean = float(np.mean(arr)) if len(arr) > 0 else 0.0
            self.std = float(np.std(arr)) if len(arr) > 0 else 1.0
            if self.std == 0.0:
                self.std = 1.0
        else:
            self.mean = normalization_stats["mean"]
            self.std = normalization_stats["std"]

        # ---- Pass 2: normalize + compute GRU-D features ----
        # We accumulate per-feature sums of normalized observed values for x_mean
        D = len(feature_names)
        feat_sum = np.zeros(D, dtype=np.float64)
        feat_cnt = np.zeros(D, dtype=np.float64)

        for times, values, masks, label in raw_records:
            # Normalize
            norm_values = []
            for v_vec, m_vec in zip(values, masks):
                norm_vec = [
                    (v - self.mean) / self.std if m > 0 else 0.0
                    for v, m in zip(v_vec, m_vec)
                ]
                norm_values.append(norm_vec)

            # Accumulate per-feature means (in normalized space)
            for nv, mv in zip(norm_values, masks):
                for d, (v, m) in enumerate(zip(nv, mv)):
                    if m > 0:
                        feat_sum[d] += v
                        feat_cnt[d] += 1

            deltas, x_lasts = compute_grud_features(times, norm_values, masks)

            self.data.append({
                "times": torch.tensor(times, dtype=torch.float32),
                "values": torch.tensor(norm_values, dtype=torch.float32),
                "masks": torch.tensor(masks, dtype=torch.float32),
                "deltas": torch.tensor(deltas, dtype=torch.float32),
                "x_lasts": torch.tensor(x_lasts, dtype=torch.float32),
            })
            self.labels.append(label)

        # Per-feature mean in normalized space (used as GRU-D imputation target)
        x_mean = np.zeros(D, dtype=np.float64)
        observed = feat_cnt > 0
        x_mean[observed] = feat_sum[observed] / feat_cnt[observed]
        self.x_mean = x_mean.astype(np.float32)

    def get_normalization_stats(self):
        return {"mean": self.mean, "std": self.std}

    def get_feature_means(self):
        return self.x_mean

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def grud_collate_fn(batch):
    data_list, label_list = zip(*batch)
    lengths = [len(d["times"]) for d in data_list]
    max_len = max(lengths)
    B = len(data_list)
    D = data_list[0]["values"].shape[-1]

    padded = {
        "times":   torch.zeros(B, max_len),
        "values":  torch.zeros(B, max_len, D),
        "masks":   torch.zeros(B, max_len, D),
        "deltas":  torch.zeros(B, max_len, D),
        "x_lasts": torch.zeros(B, max_len, D),
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }
    for i, d in enumerate(data_list):
        L = lengths[i]
        padded["times"][i, :L]   = d["times"]
        padded["values"][i, :L]  = d["values"]
        padded["masks"][i, :L]   = d["masks"]
        padded["deltas"][i, :L]  = d["deltas"]
        padded["x_lasts"][i, :L] = d["x_lasts"]

    return padded, torch.tensor(label_list, dtype=torch.float32)


# ---------------------------------------------------------------------------
# GRU-D Cell
# ---------------------------------------------------------------------------

class GRUDCell(nn.Module):
    """
    Core GRU-D cell.

    Decay equations (Che et al., 2018):
      γ_x = exp(-relu(W_γx * δ + b_γx))   — input decay  [B, D]
      γ_h = exp(-relu(W_γh * δ + b_γh))   — hidden decay [B, H]
      x̂   = m * x + (1-m) * (γ_x * x_last + (1-γ_x) * x̄)
      h̃   = γ_h ⊙ h
      h'  = GRUCell([x̂, m], h̃)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Per-feature input decay: δ ∈ R^D → γ_x ∈ (0,1]^D
        self.W_gamma_x = nn.Linear(input_dim, input_dim, bias=True)
        # Hidden state decay: δ ∈ R^D → γ_h ∈ (0,1]^H
        self.W_gamma_h = nn.Linear(input_dim, hidden_dim, bias=True)

        # GRU cell receives [x̂ (D), mask (D)] → H
        self.gru_cell = nn.GRUCell(input_size=input_dim * 2, hidden_size=hidden_dim)

        # Learnable initial hidden state
        self.h0 = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, m, delta, x_last, x_mean, h):
        """
        Args:
            x       [B, D]  observed values (zero where missing)
            m       [B, D]  binary observation mask
            delta   [B, D]  time since last observation per feature
            x_last  [B, D]  carry-forward last observed value per feature
            x_mean  [D]     per-feature empirical mean (as imputation target)
            h       [B, H]  previous hidden state
        Returns:
            h_new   [B, H]  updated hidden state
        """
        # Decay gates — clamped positive so exp gives values in (0, 1]
        gamma_x = torch.exp(-torch.relu(self.W_gamma_x(delta)))  # [B, D]
        gamma_h = torch.exp(-torch.relu(self.W_gamma_h(delta)))  # [B, H]

        # Impute missing values via decay towards empirical mean
        x_hat = m * x + (1 - m) * (gamma_x * x_last + (1 - gamma_x) * x_mean)

        # Decay hidden state
        h_decayed = gamma_h * h

        # GRU update
        h_new = self.gru_cell(torch.cat([x_hat, m], dim=-1), h_decayed)
        return h_new


# ---------------------------------------------------------------------------
# End-to-end GRU-D Model
# ---------------------------------------------------------------------------

class GRUDModel(nn.Module):
    """
    End-to-end GRU-D for binary classification.

    Encodes irregular temporal data through GRUDCell, then classifies
    with an MLP head. x_mean is registered as a non-trainable buffer.
    """

    def __init__(self, input_dim: int, hidden_dim: int, x_mean: np.ndarray):
        super().__init__()
        self.grud_cell = GRUDCell(input_dim, hidden_dim)
        self.register_buffer("x_mean", torch.tensor(x_mean, dtype=torch.float32))

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
                values  [B, T, D]
                masks   [B, T, D]
                deltas  [B, T, D]
                x_lasts [B, T, D]
                lengths [B]
        Returns:
            probs [B] — predicted probability of AKD positive
        """
        device = self.x_mean.device
        values  = batch_data["values"].to(device)
        masks   = batch_data["masks"].to(device)
        deltas  = batch_data["deltas"].to(device)
        x_lasts = batch_data["x_lasts"].to(device)
        lengths = batch_data["lengths"]

        B, T = values.shape[:2]
        h = self.grud_cell.h0.unsqueeze(0).expand(B, -1).contiguous()

        for t in range(T):
            h_new = self.grud_cell(
                values[:, t], masks[:, t], deltas[:, t],
                x_lasts[:, t], self.x_mean, h
            )
            # Zero-out update for already-completed sequences
            active = (t < lengths).to(device).float().unsqueeze(-1)
            h = active * h_new + (1 - active) * h

        return self.classifier(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, num_epochs=100,
                eval_every=5, patience=10, lr=5e-4):
    """Train with validation monitoring and early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_auc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        total_loss = 0.0
        for batch_data, labels in train_loader:
            labels = labels.to(DEVICE)
            preds = model(batch_data)
            loss = criterion(preds, labels)
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
            val_auc = roc_auc_score(val_labels, val_probs)

            avg_loss = total_loss / len(train_loader)
            print(f"    Epoch {epoch+1:3d} | loss {avg_loss:.4f} | val AUC {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    Early stopping at epoch {epoch + 1}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"    Best val AUC: {best_val_auc:.4f}")
    return model


def evaluate_model(model, loader):
    """Return (labels, probs, preds) arrays from a DataLoader."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch_data, labels in loader:
            all_probs.extend(model(batch_data).cpu().numpy())
            all_labels.extend(labels.numpy())
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs > 0.5).astype(int)
    return labels, probs, preds


# ---------------------------------------------------------------------------
# Main — cross-validation
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("GRU-D: Gated Recurrent Units with Decay (end-to-end)")
    print("=" * 80)

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)
    input_dim = len(temporal_feats)
    print(f"Temporal features: {input_dim}")

    metrics = {k: [] for k in ["auc", "auc_pr", "acc", "rec", "spec", "prec"]}

    fig, ax = plt.subplots(figsize=(9, 7))

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, seed=SEED)):
        print(f"\n{'='*80}")
        print(f"Fold {fold}")
        print("=" * 80)

        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=SEED + fold)

        # Datasets
        train_ds = GRUDDataset(train_p_obj, temporal_feats)
        stats     = train_ds.get_normalization_stats()
        x_mean    = train_ds.get_feature_means()

        val_ds  = GRUDDataset(val_p_obj,         temporal_feats, stats)
        test_ds = GRUDDataset(test_p,             temporal_feats, stats)

        print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  collate_fn=grud_collate_fn)
        val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, collate_fn=grud_collate_fn)
        test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, collate_fn=grud_collate_fn)

        # Model
        model = GRUDModel(input_dim=input_dim, hidden_dim=64, x_mean=x_mean).to(DEVICE)

        print("  Training GRU-D...")
        model = train_model(
            model, train_loader, val_loader,
            num_epochs=120, eval_every=5, patience=8, lr=5e-4
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
    ax.set_title("GRU-D: AKD Prediction (Irregular Temporal)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs("result", exist_ok=True)
    plt.savefig("result/grud.png", dpi=300)
    print("\nROC plot saved to result/grud.png")

    # Summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY — GRU-D")
    print("=" * 80)

    def stat(name, vals):
        print(f"{name:22s} | {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    stat("AUC",               metrics["auc"])
    stat("AUC-PR",            metrics["auc_pr"])
    stat("Accuracy",          metrics["acc"])
    stat("Sensitivity/Recall",metrics["rec"])
    stat("Specificity",       metrics["spec"])
    stat("Precision",         metrics["prec"])
    print("=" * 80)


if __name__ == "__main__":
    main()
