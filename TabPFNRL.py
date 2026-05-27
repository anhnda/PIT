"""
REINFORCEMENT LEARNING V5: Fixed-Variance Policy → TabPFN Judge

Key design choices:
- Policy network outputs ONLY the mean; variance is fixed at 1.0 (scaled by
  temperature).  This removes noisy learned-std that RL could not optimise.
- Adaptive RL guard: revert to pretrained weights if RL degrades performance.
- Enriched temporal statistics (last/mean/std/min/max/slope) instead of last-only.
- Feature selection via mutual information (top-80).
- Pretrained/RL ensemble blend when RL improves performance.
- TabPFN with n_estimators=16.

Shared building blocks (SimpleStaticEncoder, HybridDataset, hybrid_collate_fn,
select_features_mi) live in utils/rl_common.py.
"""

import sys
import os
import copy
import math
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, auc,
    confusion_matrix,
)

os.environ["SEGMENT_WRITE_KEY"]          = ""
os.environ["ANALYTICS_WRITE_KEY"]        = ""
os.environ["TABPFN_DISABLE_ANALYTICS"]   = "1"

try:
    import analytics
    analytics.write_key = None
    analytics.disable()
except Exception:
    pass

from tabpfn import TabPFNClassifier

PT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PT)

from utils.rl_common import (
    FIXED_FEATURES, seed_everything,
    SimpleStaticEncoder, HybridDataset, hybrid_collate_fn,
    select_features_mi,
)
from TimeEmbedding import DEVICE, TimeEmbeddedRNNCell
from TimeEmbeddingVal import (
    get_all_temporal_features, extract_temporal_data,
    load_and_prepare_patients, split_patients_train_val,
)
from utils.prepare_data import trainTestPatients

xseed = 42
seed_everything(xseed)


# ==============================================================================
# RNN Policy Network — Fixed Unit Variance
# ==============================================================================

class RNNPolicyNetwork(nn.Module):
    """RNN policy that outputs ONLY the mean; variance is fixed at 1.0.

    During stochastic mode: z = mean + temperature * ε, ε ~ N(0, I)
    During deterministic mode: z = mean

    The fixed variance removes the noisy learned log-std and simplifies the
    REINFORCE gradient to a pure mean-update signal.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, time_dim=32):
        super().__init__()
        self.rnn_cell   = TimeEmbeddedRNNCell(input_dim, hidden_dim, time_dim)
        self.fc_mean    = nn.Linear(hidden_dim, latent_dim)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

    def forward(self, batch_data, deterministic=False, temperature=0.05):
        """
        Returns
        -------
        z        : sampled (or mean) latent vector  [batch, latent_dim]
        log_prob : per-sample log-probability       [batch]  (None if deterministic)
        mean     : distribution mean                [batch, latent_dim]
        """
        times   = batch_data["times"].to(DEVICE)
        values  = batch_data["values"].to(DEVICE)
        masks   = batch_data["masks"].to(DEVICE)
        lengths = batch_data["lengths"].to(DEVICE)

        h    = self.rnn_cell(times, values, masks, lengths)
        mean = self.fc_mean(h)

        if deterministic:
            return mean, None, mean

        eps      = torch.randn_like(mean)
        z        = mean + temperature * eps
        diff     = z.detach() - mean
        log_prob = -0.5 * (diff ** 2).sum(dim=-1) / (temperature ** 2)
        return z, log_prob, mean


# ==============================================================================
# Supervised Classification Head (used during pretraining)
# ==============================================================================

class SupervisedHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1     = nn.Linear(input_dim, hidden_dim)
        self.bn1     = nn.BatchNorm1d(hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2     = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3     = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))


# ==============================================================================
# Enhanced Pretraining
# ==============================================================================

def pretrain_rnn_enhanced(policy_net, train_loader, val_loader, epochs=50):
    """Supervised pretraining to give the RNN a good starting point."""
    print("  [Pretraining] Supervised warm-up...")

    supervised_head = SupervisedHead(
        policy_net.latent_dim + len(FIXED_FEATURES)
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(supervised_head.parameters()),
        lr=0.001,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    criterion = nn.BCELoss()

    best_aupr  = 0.0
    best_state = None
    patience   = 12
    counter    = 0

    for epoch in range(epochs):
        policy_net.train()
        supervised_head.train()

        for t_data, labels, s_data in train_loader:
            labels = labels.to(DEVICE)
            s_data = s_data.to(DEVICE)
            z, _, _ = policy_net(t_data, deterministic=True)
            preds   = supervised_head(torch.cat([z, s_data], dim=1)).squeeze(-1)
            loss    = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy_net.parameters()) + list(supervised_head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

        if (epoch + 1) % 3 == 0:
            policy_net.eval()
            supervised_head.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for t_data, labels, s_data in val_loader:
                    s_data = s_data.to(DEVICE)
                    z, _, _ = policy_net(t_data, deterministic=True)
                    preds   = supervised_head(torch.cat([z, s_data], dim=1)).squeeze(-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

            val_aupr = average_precision_score(all_labels, all_preds)
            scheduler.step(val_aupr)
            print(f"    Pretrain Epoch {epoch+1} | Val AUPR: {val_aupr:.4f}")

            if val_aupr > best_aupr:
                best_aupr  = val_aupr
                best_state = copy.deepcopy(policy_net.state_dict())
                counter    = 0
            else:
                counter += 1
                if counter >= patience:
                    break

    if best_state is not None:
        policy_net.load_state_dict(best_state)
    print(f"  [Pretraining] Done. Best Val AUPR: {best_aupr:.4f}")
    return policy_net


# ==============================================================================
# Enriched Feature Extraction
# ==============================================================================

def extract_enriched_features_and_logprobs(
    policy_net, loader, deterministic=False, temperature=1.0
):
    """Extract [Static + Last + Mean + Std + Min + Max + Slope + Z] features.

    The six temporal statistics per feature replace the last-value-only
    representation used in XGRL / CatBoostRL, giving TabPFN richer signal.
    """
    if deterministic:
        policy_net.eval()
    else:
        policy_net.train()

    all_features  = []
    all_labels    = []
    all_log_probs = []

    with torch.set_grad_enabled(not deterministic):
        for t_data, labels, s_data in loader:
            z, log_prob, mean = policy_net(
                t_data, deterministic=deterministic, temperature=temperature
            )
            z_np = (mean if deterministic else z).detach().cpu().numpy()

            vals  = t_data["values"].cpu().numpy()
            masks = t_data["masks"].cpu().numpy()
            n_feats = vals.shape[2]

            batch_stats = {k: [] for k in ["last", "mean", "std", "min", "max", "slope"]}

            for i in range(len(vals)):
                row = {k: [] for k in batch_stats}
                for f_idx in range(n_feats):
                    f_vals    = vals[i, :, f_idx]
                    f_mask    = masks[i, :, f_idx]
                    valid_idx = np.where(f_mask > 0)[0]
                    if len(valid_idx) > 0:
                        vv = f_vals[valid_idx]
                        row["last"].append(vv[-1])
                        row["mean"].append(np.mean(vv))
                        row["std"].append(np.std(vv) if len(vv) > 1 else 0.0)
                        row["min"].append(np.min(vv))
                        row["max"].append(np.max(vv))
                        row["slope"].append(vv[-1] - vv[0] if len(vv) > 1 else 0.0)
                    else:
                        for k in row:
                            row[k].append(0.0)
                for k in batch_stats:
                    batch_stats[k].append(row[k])

            s_np = s_data.numpy()
            combined = np.hstack(
                [s_np]
                + [np.array(batch_stats[k]) for k in ["last", "mean", "std", "min", "max", "slope"]]
                + [z_np]
            )
            all_features.append(combined)
            all_labels.extend(labels.numpy())

            if not deterministic and log_prob is not None:
                all_log_probs.append(log_prob)

    features  = np.vstack(all_features)
    labels    = np.array(all_labels)
    log_probs = torch.cat(all_log_probs) if all_log_probs else None

    return features, labels, log_probs


# ==============================================================================
# TabPFN Evaluation
# ==============================================================================

def evaluate_with_tabpfn(policy_net, train_loader, val_loader, tabpfn_params, feat_indices=None):
    policy_net.eval()
    with torch.no_grad():
        X_train, y_train, _ = extract_enriched_features_and_logprobs(
            policy_net, train_loader, deterministic=True
        )
        X_val, y_val, _ = extract_enriched_features_and_logprobs(
            policy_net, val_loader, deterministic=True
        )

    if feat_indices is not None:
        X_train = X_train[:, feat_indices]
        X_val   = X_val[:, feat_indices]

    model = TabPFNClassifier(**tabpfn_params)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_val)[:, 1]

    val_auc  = roc_auc_score(y_val, y_proba)
    val_aupr = average_precision_score(y_val, y_proba)
    return val_auc, val_aupr


# ==============================================================================
# Adaptive RL Training
# ==============================================================================

def train_policy_adaptive_rl(
    policy_net,
    train_loader,
    val_loader,
    tabpfn_params,
    epochs=40,
    update_tabpfn_every=5,
    top_k_features=80,
):
    """RL fine-tuning with adaptive guard: reverts if RL degrades performance.

    Returns
    -------
    policy_net   : trained (or reverted) policy
    feat_indices : MI-selected feature indices
    was_reverted : bool — True if the pretrained weights were restored
    """
    pre_rl_state = copy.deepcopy(policy_net.state_dict())

    policy_net.eval()
    with torch.no_grad():
        X_train_pre, y_train_pre, _ = extract_enriched_features_and_logprobs(
            policy_net, train_loader, deterministic=True
        )
        X_val_pre, y_val_pre, _ = extract_enriched_features_and_logprobs(
            policy_net, val_loader, deterministic=True
        )

    _, _, feat_indices = select_features_mi(
        X_train_pre, y_train_pre, X_val_pre, top_k=top_k_features
    )

    pre_rl_auc, pre_rl_aupr = evaluate_with_tabpfn(
        policy_net, train_loader, val_loader, tabpfn_params, feat_indices
    )
    print(f"  [Pre-RL Baseline] AUC: {pre_rl_auc:.4f} | AUPR: {pre_rl_aupr:.4f}")

    optimizer  = torch.optim.Adam(policy_net.parameters(), lr=5e-5)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_aupr      = pre_rl_aupr
    best_state         = copy.deepcopy(policy_net.state_dict())
    patience           = 15
    patience_counter   = 0
    degradation_counter = 0

    print("  [Adaptive RL] Fixed-variance policy, AUPR-focused...")

    tabpfn_model = None

    for epoch in range(epochs):
        temperature = max(0.3, 0.7 - epoch / (epochs * 0.8))

        policy_net.train()
        X_train, y_train, log_probs_train = extract_enriched_features_and_logprobs(
            policy_net, train_loader, deterministic=False, temperature=temperature
        )
        X_train_sel = X_train[:, feat_indices]

        if epoch % update_tabpfn_every == 0 or tabpfn_model is None:
            tabpfn_model = TabPFNClassifier(**tabpfn_params)
            tabpfn_model.fit(X_train_sel, y_train)

        y_train_proba = tabpfn_model.predict_proba(X_train_sel)[:, 1]

        rewards = np.where(
            (y_train == 1) & (y_train_proba > 0.5), y_train_proba * 2.0,
            np.where(
                y_train == 1, y_train_proba * 0.5,
                np.where(y_train_proba < 0.5, 1 - y_train_proba, (1 - y_train_proba) * 0.5),
            ),
        )

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        log_probs_train = log_probs_train.to(DEVICE)
        policy_loss     = -(log_probs_train * rewards_tensor).mean()

        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.3)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            val_auc, val_aupr = evaluate_with_tabpfn(
                policy_net, train_loader, val_loader, tabpfn_params, feat_indices
            )
            current_lr = scheduler.get_last_lr()[0]
            print(f"    Epoch {epoch+1:3d} | Temp: {temperature:.3f} | LR: {current_lr:.2e} | "
                  f"Val AUC: {val_auc:.4f} | Val AUPR: {val_aupr:.4f}")

            if val_aupr > best_val_aupr:
                best_val_aupr       = val_aupr
                best_state          = copy.deepcopy(policy_net.state_dict())
                patience_counter    = 0
                degradation_counter = 0
            else:
                patience_counter += 1
                if val_aupr < pre_rl_aupr - 0.005:
                    degradation_counter += 1
                    if degradation_counter >= 3:
                        print("    RL degrading below pretrained. Reverting.")
                        policy_net.load_state_dict(pre_rl_state)
                        return policy_net, feat_indices, True
                else:
                    degradation_counter = 0

                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

    if best_val_aupr > pre_rl_aupr:
        policy_net.load_state_dict(best_state)
        print(f"  [RL] Best AUPR: {best_val_aupr:.4f} (pre-RL: {pre_rl_aupr:.4f})")
        return policy_net, feat_indices, False
    else:
        policy_net.load_state_dict(pre_rl_state)
        print(f"  [RL] No improvement. Reverted to pretrained (pre-RL: {pre_rl_aupr:.4f})")
        return policy_net, feat_indices, True


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 80)
    print("RL POLICY V5 (Fixed Variance + Adaptive) — TabPFN Judge")
    print("=" * 80)

    patients       = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)

    encoder = SimpleStaticEncoder(FIXED_FEATURES)
    encoder.fit(patients.patientList)
    print(f"Input: {len(temporal_feats)} Temporal + {len(FIXED_FEATURES)} Static Features")

    metrics_rl    = {k: [] for k in ["auc", "auc_pr"]}
    reverted_folds = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, seed=xseed)):
        print(f"\n{'='*80}\nFold {fold}\n{'='*80}")

        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42 + fold)
        train_p     = train_p_obj.patientList
        val_p       = val_p_obj.patientList
        test_p_list = test_p.patientList

        train_ds = HybridDataset(train_p,       temporal_feats, encoder)
        stats    = train_ds.get_normalization_stats()
        val_ds   = HybridDataset(val_p,          temporal_feats, encoder, stats)
        test_ds  = HybridDataset(test_p_list,    temporal_feats, encoder, stats)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  collate_fn=hybrid_collate_fn)
        val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
        test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)

        latent_dim = 12
        policy_net = RNNPolicyNetwork(
            input_dim=len(temporal_feats), hidden_dim=64,
            latent_dim=latent_dim, time_dim=32,
        ).to(DEVICE)

        tabpfn_params = {
            "device":       "cuda" if torch.cuda.is_available() else "cpu",
            "n_estimators": 16,
        }

        # Step 1: Enhanced pretraining
        policy_net = pretrain_rnn_enhanced(policy_net, train_loader, val_loader, epochs=50)
        pretrained_state = copy.deepcopy(policy_net.state_dict())

        # Step 2: Adaptive RL fine-tuning
        policy_net, feat_indices, was_reverted = train_policy_adaptive_rl(
            policy_net, train_loader, val_loader, tabpfn_params,
            epochs=40, update_tabpfn_every=5, top_k_features=80,
        )
        if was_reverted:
            reverted_folds.append(fold)

        # Step 3: Final evaluation with ensemble blend
        print("\n  [Final Test Evaluation]")
        policy_net.eval()
        with torch.no_grad():
            X_train_rl, y_train_f, _ = extract_enriched_features_and_logprobs(
                policy_net, train_loader, deterministic=True
            )
            X_test_rl, y_test_f, _ = extract_enriched_features_and_logprobs(
                policy_net, test_loader, deterministic=True
            )

        if not was_reverted:
            pretrained_net = RNNPolicyNetwork(
                input_dim=len(temporal_feats), hidden_dim=64,
                latent_dim=latent_dim, time_dim=32,
            ).to(DEVICE)
            pretrained_net.load_state_dict(pretrained_state)
            pretrained_net.eval()
            with torch.no_grad():
                X_train_pre, _, _ = extract_enriched_features_and_logprobs(
                    pretrained_net, train_loader, deterministic=True
                )
                X_test_pre, _, _ = extract_enriched_features_and_logprobs(
                    pretrained_net, test_loader, deterministic=True
                )
            X_train_f = 0.5 * X_train_pre + 0.5 * X_train_rl
            X_test_f  = 0.5 * X_test_pre  + 0.5 * X_test_rl
        else:
            X_train_f = X_train_rl
            X_test_f  = X_test_rl

        X_train_sel = X_train_f[:, feat_indices]
        X_test_sel  = X_test_f[:, feat_indices]

        final_tabpfn = TabPFNClassifier(**tabpfn_params)
        final_tabpfn.fit(X_train_sel, y_train_f)
        y_test_proba = final_tabpfn.predict_proba(X_test_sel)[:, 1]

        prec, rec, _ = precision_recall_curve(y_test_f, y_test_proba)
        fold_auc  = roc_auc_score(y_test_f, y_test_proba)
        fold_aupr = auc(rec, prec)

        metrics_rl["auc"].append(fold_auc)
        metrics_rl["auc_pr"].append(fold_aupr)

        fpr, tpr, _ = roc_curve(y_test_f, y_test_proba)
        ax1.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC={fold_auc:.3f})")

        print(f"  Test AUC: {fold_auc:.4f} | Test AUPR: {fold_aupr:.4f}")

    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.set_title("ROC Curves (RL V5 — Fixed Variance)")
    ax1.legend(fontsize=8)

    n_folds = len(metrics_rl["auc_pr"])
    ax2.bar(range(n_folds), metrics_rl["auc_pr"], alpha=0.7, label="AUPR")
    ax2.set_xlabel("Fold"); ax2.set_ylabel("AUPR")
    ax2.set_title("AUPR per Fold"); ax2.legend()

    plt.tight_layout()
    os.makedirs("result", exist_ok=True)
    plt.savefig("result/tabpfn_rl_v5.png", dpi=150)
    plt.close()

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    def print_stat(name, vals):
        print(f"{name:15s} | {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print_stat("AUC",    metrics_rl["auc"])
    print_stat("AUC-PR", metrics_rl["auc_pr"])

    if reverted_folds:
        print(f"\nFolds reverted to pretrained (RL degraded): {reverted_folds}")
    else:
        print("\nAll folds improved with RL.")


if __name__ == "__main__":
    main()
