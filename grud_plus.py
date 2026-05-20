"""
GRU-D + Tabular Classifier (GRU-D Plus)

Pipeline:
  1. Pre-train GRU-D encoder end-to-end (BCE loss + validation early-stopping)
  2. Freeze encoder; extract Z = encoder.encode(temporal_data)        [H]
  3. Concatenate: [static_features | last_observed_values | Z]
  4. Fit tabular classifier on combined features → evaluate

Classifier options (--clf):
  tabpfn   (default) — TabPFNClassifier (≤1024 train samples, ≤100 features)
  xgboost            — XGBClassifier
  catboost           — CatBoostClassifier

Usage:
  python grud_plus.py                    # TabPFN
  python grud_plus.py --clf xgboost
  python grud_plus.py --clf catboost
"""

import os
import sys
import copy
import random
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, auc,
)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PT)

os.environ.setdefault("SEGMENT_WRITE_KEY", "")
os.environ.setdefault("ANALYTICS_WRITE_KEY", "")
os.environ.setdefault("TABPFN_DISABLE_ANALYTICS", "1")

from utils.prepare_data import trainTestPatients
from TimeEmbedding import DEVICE, get_all_temporal_features, extract_temporal_data, load_and_prepare_patients
from TimeEmbeddingVal import split_patients_train_val
from grud import GRUDCell, compute_grud_features

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

FIXED_FEATURES = [
    "age", "gender", "race", "chronic_pulmonary_disease", "ckd_stage",
    "congestive_heart_failure", "dka_type", "history_aci", "history_ami",
    "hypertension", "liver_disease", "macroangiopathy", "malignant_cancer",
    "microangiopathy", "uti", "oasis", "saps2", "sofa",
    "mechanical_ventilation", "use_NaHCO3", "preiculos", "gcs_unable",
]

# ---------------------------------------------------------------------------
# Static encoder (categorical → numeric label encoding)
# ---------------------------------------------------------------------------

class SimpleStaticEncoder:
    def __init__(self, features):
        self.features = features
        self.mappings = {f: {} for f in features}
        self.counts   = {f: 0  for f in features}

    def fit(self, patients):
        for p in patients:
            for f in self.features:
                val = p.measures.get(f, 0.0)
                if hasattr(val, "values"):
                    val = list(val.values())[0] if len(val) > 0 else 0.0
                val_str = str(val)
                try:
                    float(val)
                except ValueError:
                    if val_str not in self.mappings[f]:
                        self.mappings[f][val_str] = float(self.counts[f])
                        self.counts[f] += 1

    def transform(self, patient):
        vec = []
        for f in self.features:
            val = patient.measures.get(f, 0.0)
            if hasattr(val, "values"):
                val = list(val.values())[0] if len(val) > 0 else 0.0
            try:
                vec.append(float(val))
            except ValueError:
                vec.append(self.mappings[f].get(str(val), -1.0))
        return vec


# ---------------------------------------------------------------------------
# Hybrid dataset: GRU-D temporal fields + static features
# ---------------------------------------------------------------------------

class HybridGRUDDataset(Dataset):
    """
    Stores per-patient:
      times, values, masks, deltas, x_lasts  — GRU-D inputs
      static                                  — fixed clinical / demographic
      label                                   — AKD positive/negative
    """

    def __init__(self, patients, feature_names, static_encoder,
                 normalization_stats=None):
        self.feature_names = feature_names
        self.data          = []
        self.static_data   = []
        self.labels        = []

        patient_list = patients.patientList if hasattr(patients, "patientList") else patients
        all_observed = []

        # ---- Pass 1: collect raw data ----
        raw = []
        for patient in patient_list:
            times, values, masks = extract_temporal_data(patient, feature_names)
            if times is None:
                continue
            s_vec = static_encoder.transform(patient)
            raw.append((times, values, masks, s_vec, 1 if patient.akdPositive else 0))
            for v_vec, m_vec in zip(values, masks):
                for v, m in zip(v_vec, m_vec):
                    if m > 0:
                        all_observed.append(v)

        # ---- Normalization stats ----
        if normalization_stats is None:
            arr = np.array(all_observed)
            self.mean = float(np.mean(arr)) if len(arr) > 0 else 0.0
            self.std  = float(np.std(arr))  if len(arr) > 0 else 1.0
            if self.std == 0.0:
                self.std = 1.0
        else:
            self.mean = normalization_stats["mean"]
            self.std  = normalization_stats["std"]

        # ---- Pass 2: normalize + compute GRU-D features ----
        D = len(feature_names)
        feat_sum = np.zeros(D, np.float64)
        feat_cnt = np.zeros(D, np.float64)

        for times, values, masks, s_vec, label in raw:
            norm_vals = [
                [(v - self.mean) / self.std if m > 0 else 0.0 for v, m in zip(vv, mv)]
                for vv, mv in zip(values, masks)
            ]
            for nv, mv in zip(norm_vals, masks):
                for d, (v, m) in enumerate(zip(nv, mv)):
                    if m > 0:
                        feat_sum[d] += v
                        feat_cnt[d] += 1

            deltas, x_lasts = compute_grud_features(times, norm_vals, masks)

            self.data.append({
                "times":   torch.tensor(times,     dtype=torch.float32),
                "values":  torch.tensor(norm_vals, dtype=torch.float32),
                "masks":   torch.tensor(masks,     dtype=torch.float32),
                "deltas":  torch.tensor(deltas,    dtype=torch.float32),
                "x_lasts": torch.tensor(x_lasts,   dtype=torch.float32),
            })
            self.static_data.append(torch.tensor(s_vec, dtype=torch.float32))
            self.labels.append(label)

        self.x_mean = np.where(feat_cnt > 0, feat_sum / feat_cnt, 0.0).astype(np.float32)

    def get_normalization_stats(self):
        return {"mean": self.mean, "std": self.std}

    def get_feature_means(self):
        return self.x_mean

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.static_data[idx]


def hybrid_grud_collate_fn(batch):
    data_list, label_list, static_list = zip(*batch)
    lengths = [len(d["times"]) for d in data_list]
    max_len = max(lengths)
    B, D = len(data_list), data_list[0]["values"].shape[-1]

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
        for k in ("times", "values", "masks", "deltas", "x_lasts"):
            padded[k][i, :L] = d[k]

    return padded, torch.tensor(label_list, dtype=torch.float32), torch.stack(static_list)


# ---------------------------------------------------------------------------
# GRU-D Encoder  (same dynamics as grud.py, adds encode() method)
# ---------------------------------------------------------------------------

class GRUDEncoder(nn.Module):
    """
    GRU-D with encode() exposing the hidden state before the MLP head.

    forward()  → P(AKD)    [used during pre-training]
    encode()   → h  [B, H] [used for feature extraction]
    """

    def __init__(self, input_dim: int, hidden_dim: int, x_mean: np.ndarray):
        super().__init__()
        self.grud_cell = GRUDCell(input_dim, hidden_dim)
        self.register_buffer("x_mean", torch.tensor(x_mean, dtype=torch.float32))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def encode(self, batch_data: dict) -> torch.Tensor:
        """Return final hidden state [B, H]."""
        device  = self.x_mean.device
        values  = batch_data["values"].to(device)
        masks   = batch_data["masks"].to(device)
        deltas  = batch_data["deltas"].to(device)
        x_lasts = batch_data["x_lasts"].to(device)
        lengths = batch_data["lengths"]

        B, T = values.shape[:2]
        h = self.grud_cell.h0.unsqueeze(0).expand(B, -1).contiguous()

        for t in range(T):
            h_new  = self.grud_cell(values[:, t], masks[:, t], deltas[:, t],
                                    x_lasts[:, t], self.x_mean, h)
            active = (t < lengths).to(device).float().unsqueeze(-1)
            h = active * h_new + (1.0 - active) * h

        return h  # [B, H]

    def forward(self, batch_data: dict) -> torch.Tensor:
        return self.classifier(self.encode(batch_data)).squeeze(-1)


# ---------------------------------------------------------------------------
# Encoder pre-training
# ---------------------------------------------------------------------------

def pretrain_encoder(encoder: GRUDEncoder,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     num_epochs: int = 100,
                     eval_every: int = 5,
                     patience: int = 8,
                     lr: float = 5e-4) -> GRUDEncoder:
    """Pre-train encoder end-to-end with BCE loss and early stopping."""
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    criterion = nn.BCELoss()
    best_auc, best_state, no_improve = 0.0, None, 0

    for epoch in range(num_epochs):
        encoder.train()
        total_loss = 0.0
        for batch_data, labels, _ in train_loader:   # ignore static here
            labels = labels.to(DEVICE)
            loss   = criterion(encoder(batch_data), labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % eval_every == 0:
            encoder.eval()
            probs, truths = [], []
            with torch.no_grad():
                for batch_data, labels, _ in val_loader:
                    probs.extend(encoder(batch_data).cpu().numpy())
                    truths.extend(labels.numpy())
            val_auc = roc_auc_score(truths, probs)
            print(f"    [pretrain] epoch {epoch+1:3d} | "
                  f"loss {total_loss/len(train_loader):.4f} | val AUC {val_auc:.4f}")
            if val_auc > best_auc:
                best_auc   = val_auc
                best_state = copy.deepcopy(encoder.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    [pretrain] early stop at epoch {epoch + 1}")
                    break

    if best_state is not None:
        encoder.load_state_dict(best_state)
        print(f"    [pretrain] best val AUC: {best_auc:.4f}")
    return encoder


# ---------------------------------------------------------------------------
# Feature extraction: [static | last_values | z]
# ---------------------------------------------------------------------------

def extract_combined_features(encoder: GRUDEncoder,
                               loader: DataLoader) -> tuple:
    """
    Returns (X, y) where X = [static | last_observed | z].

    last_observed: last non-missing normalized value per temporal feature.
    z: GRU-D hidden state (encoder.encode()).
    """
    encoder.eval()
    all_X, all_y = [], []

    with torch.no_grad():
        for batch_data, labels, static in loader:
            # Encoder hidden state
            z = encoder.encode(batch_data).cpu().numpy()   # [B, H]

            # Last observed value per feature
            vals    = batch_data["values"].numpy()          # [B, T, D]
            masks   = batch_data["masks"].numpy()           # [B, T, D]
            lengths = batch_data["lengths"].numpy()         # [B]
            B, T, D = vals.shape
            last_vals = np.zeros((B, D), dtype=np.float32)
            for i in range(B):
                for d in range(D):
                    obs = np.where(masks[i, :lengths[i], d] > 0)[0]
                    if len(obs) > 0:
                        last_vals[i, d] = vals[i, obs[-1], d]

            s_np = static.numpy()                           # [B, S]
            X    = np.hstack([s_np, last_vals, z])          # [B, S+D+H]
            all_X.append(X)
            all_y.extend(labels.numpy())

    return np.vstack(all_X), np.array(all_y)


# ---------------------------------------------------------------------------
# Tabular classifier factory
# ---------------------------------------------------------------------------

def build_classifier(clf_type: str, pos_weight: float = 1.0):
    clf_type = clf_type.lower()
    if clf_type == "tabpfn":
        try:
            from tabpfn import TabPFNClassifier
            return TabPFNClassifier(device="cpu")
        except ImportError:
            raise ImportError("tabpfn not installed: pip install tabpfn")
    elif clf_type == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=pos_weight, subsample=0.8,
            colsample_bytree=0.8, eval_metric="auc", random_state=SEED,
        )
    elif clf_type == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=SEED, verbose=False,
        )
    else:
        raise ValueError(f"Unknown classifier: {clf_type}. "
                         "Choose from tabpfn, xgboost, catboost.")


def fit_classifier(clf, clf_type: str, X_train: np.ndarray, y_train: np.ndarray):
    """Fit with TabPFN sample / feature cap handled automatically."""
    if clf_type == "tabpfn":
        max_samples  = 1024
        max_features = 100
        # Feature cap: take first max_features columns if needed
        if X_train.shape[1] > max_features:
            print(f"    [tabpfn] truncating features {X_train.shape[1]} → {max_features}")
            X_train = X_train[:, :max_features]
        # Sample cap: stratified subsample
        if len(X_train) > max_samples:
            print(f"    [tabpfn] subsampling {len(X_train)} → {max_samples} (stratified)")
            pos = np.where(y_train == 1)[0]
            neg = np.where(y_train == 0)[0]
            n_pos = int(max_samples * len(pos) / len(y_train))
            n_neg = max_samples - n_pos
            rng   = np.random.default_rng(SEED)
            idx   = np.concatenate([
                rng.choice(pos, min(n_pos, len(pos)), replace=False),
                rng.choice(neg, min(n_neg, len(neg)), replace=False),
            ])
            X_train, y_train = X_train[idx], y_train[idx]
        clf.fit(X_train, y_train)
        return clf, max_features   # return cap so test can be trimmed too
    else:
        clf.fit(X_train, y_train)
        return clf, None


# ---------------------------------------------------------------------------
# Main — cross-validation
# ---------------------------------------------------------------------------

def main(clf_type: str = "tabpfn"):
    print("=" * 80)
    print(f"GRU-D Plus → {clf_type.upper()} | [Static + Last + GRU-D encoding]")
    print("=" * 80)

    patients       = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)
    input_dim      = len(temporal_feats)
    print(f"Temporal features: {input_dim} | Static features: {len(FIXED_FEATURES)}")

    static_enc = SimpleStaticEncoder(FIXED_FEATURES)
    static_enc.fit(patients.patientList)

    metrics = {k: [] for k in ["auc", "auc_pr", "acc", "rec", "spec", "prec"]}
    fig, ax = plt.subplots(figsize=(9, 7))

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, seed=SEED)):
        print(f"\n{'='*80}\nFold {fold}\n{'='*80}")

        train_p_obj, val_p_obj = split_patients_train_val(
            train_full, val_ratio=0.1, seed=SEED + fold
        )

        # Datasets
        train_ds = HybridGRUDDataset(train_p_obj, temporal_feats, static_enc)
        stats    = train_ds.get_normalization_stats()
        x_mean   = train_ds.get_feature_means()
        val_ds   = HybridGRUDDataset(val_p_obj, temporal_feats, static_enc, stats)
        test_ds  = HybridGRUDDataset(test_p,    temporal_feats, static_enc, stats)
        print(f"  Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                  collate_fn=hybrid_grud_collate_fn)
        val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                                  collate_fn=hybrid_grud_collate_fn)
        test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False,
                                  collate_fn=hybrid_grud_collate_fn)

        # ---- Step 1: pre-train GRU-D encoder ----
        encoder = GRUDEncoder(input_dim, hidden_dim=64, x_mean=x_mean).to(DEVICE)
        print("  Pre-training GRU-D encoder...")
        encoder = pretrain_encoder(encoder, train_loader, val_loader,
                                   num_epochs=120, eval_every=5, patience=8)

        # ---- Step 2: extract combined features ----
        print("  Extracting combined features [static | last | z]...")
        encoder.eval()
        X_train, y_train = extract_combined_features(encoder, train_loader)
        X_test,  y_test  = extract_combined_features(encoder, test_loader)
        feat_dim = X_train.shape[1]
        print(f"  Combined feature dim: {feat_dim}  "
              f"(static={len(FIXED_FEATURES)} | last={input_dim} | z=64)")

        # ---- Step 3: fit tabular classifier ----
        pos_weight = float(np.sum(y_train == 0)) / max(1, np.sum(y_train == 1))
        clf        = build_classifier(clf_type, pos_weight)
        print(f"  Fitting {clf_type.upper()} classifier...")

        clf, feat_cap = fit_classifier(clf, clf_type, X_train.copy(), y_train.copy())

        # Apply same feature cap to test if needed
        X_test_clf = X_test[:, :feat_cap] if feat_cap else X_test
        y_prob = clf.predict_proba(X_test_clf)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        # ---- Metrics ----
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)

        fold_auc  = roc_auc_score(y_test, y_prob)
        fold_aupr = auc(rec_arr, prec_arr)
        fold_acc  = accuracy_score(y_test, y_pred)
        fold_rec  = recall_score(y_test, y_pred, zero_division=0)
        fold_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fold_prec = precision_score(y_test, y_pred, zero_division=0)

        for k, v in zip(["auc","auc_pr","acc","rec","spec","prec"],
                        [fold_auc, fold_aupr, fold_acc, fold_rec, fold_spec, fold_prec]):
            metrics[k].append(v)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC={fold_auc:.3f})")
        print(f"  Fold {fold} → AUC {fold_auc:.4f} | AUPR {fold_aupr:.4f} | "
              f"Sens {fold_rec:.4f} | Spec {fold_spec:.4f}")

    # ROC plot
    ax.plot([0, 1], [0, 1], "--", color="navy", lw=2, label="Random")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"GRU-D Plus ({clf_type.upper()}): AKD Prediction",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs("result", exist_ok=True)
    out_path = f"result/grud_plus_{clf_type}.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nROC plot saved to {out_path}")

    # Summary
    print("\n" + "=" * 80)
    print(f"FINAL RESULTS — GRU-D Plus ({clf_type.upper()})")
    print("=" * 80)
    for name, key in [("AUC","auc"),("AUC-PR","auc_pr"),("Accuracy","acc"),
                      ("Sensitivity/Recall","rec"),("Specificity","spec"),("Precision","prec")]:
        vals = metrics[key]
        print(f"{name:22s} | {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRU-D Plus: temporal encoder + tabular classifier")
    parser.add_argument("--clf", default="tabpfn",
                        choices=["tabpfn", "xgboost", "catboost"],
                        help="Tabular classifier (default: tabpfn)")
    args = parser.parse_args()
    main(clf_type=args.clf)
