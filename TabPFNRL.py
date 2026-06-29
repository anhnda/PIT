"""
REINFORCEMENT LEARNING V3: RNN Policy Network → TabPFN Judge

Goal: Beat baseline on BOTH AUC and AUPR

Key improvements over V2:
1. Enhanced Pretraining: Better initialization with longer training
2. AUPR-Focused Rewards: Directly optimize AUPR metric
3. Conservative RL: Lower temperature, smaller learning rate
4. Feature Enrichment: Add more temporal statistics beyond last value
5. Ensemble Sampling: Multiple policy samples for robustness
"""

import pandas as pd
import numpy as np
import sys
import copy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
import random
import os
import math

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc,
)

os.environ["SEGMENT_WRITE_KEY"] = ""
os.environ["ANALYTICS_WRITE_KEY"] = ""
os.environ["TABPFN_DISABLE_ANALYTICS"] = "1"

try:
    import analytics
    analytics.write_key = None
    analytics.disable()
except Exception:
    pass

from tabpfn import TabPFNClassifier

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

xseed = 42
seed_everything(xseed)

PT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PT)

from constants import NULLABLE_MEASURES
from utils.class_patient import Patients
from utils.prepare_data import trainTestPatients, encodeCategoricalData
from TimeEmbeddingVal import (
    get_all_temporal_features,
    extract_temporal_data,
    load_and_prepare_patients,
    split_patients_train_val,
)
from TimeEmbedding import DEVICE, TimeEmbeddedRNNCell

FIXED_FEATURES = [
    "age", "gender", "race", "chronic_pulmonary_disease", "ckd_stage",
    "congestive_heart_failure", "dka_type", "history_aci", "history_ami",
    "hypertension", "liver_disease", "macroangiopathy", "malignant_cancer",
    "microangiopathy", "uti", "oasis", "saps2", "sofa",
    "mechanical_ventilation", "use_NaHCO3", "preiculos", "gcs_unable"
]

# ==============================================================================
# 1. Static Encoder
# ==============================================================================

class SimpleStaticEncoder:
    """Encodes categorical static features"""
    def __init__(self, features):
        self.features = features
        self.mappings = {f: {} for f in features}
        self.counts = {f: 0 for f in features}

    def fit(self, patients):
        for p in patients:
            for f in self.features:
                val = p.measures.get(f, 0.0)
                if hasattr(val, 'values') and len(val) > 0:
                    val = list(val.values())[0]
                elif hasattr(val, 'values'):
                    val = 0.0

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
            if hasattr(val, 'values') and len(val) > 0:
                val = list(val.values())[0]
            elif hasattr(val, 'values'):
                val = 0.0

            try:
                numeric_val = float(val)
            except ValueError:
                numeric_val = self.mappings[f].get(str(val), -1.0)
            vec.append(numeric_val)
        return vec

# ==============================================================================
# 2. Dataset
# ==============================================================================

class HybridDataset(Dataset):
    def __init__(self, patients, feature_names, static_encoder, normalization_stats=None):
        self.data = []
        self.labels = []
        self.static_data = []
        self.feature_names = feature_names

        all_values = []
        patient_list = patients.patientList if hasattr(patients, 'patientList') else patients

        for patient in patient_list:
            times, values, masks = extract_temporal_data(patient, feature_names)
            if times is None:
                continue

            s_vec = static_encoder.transform(patient)
            self.static_data.append(torch.tensor(s_vec, dtype=torch.float32))
            self.data.append({'times': times, 'values': values, 'masks': masks})
            self.labels.append(1 if patient.akdPositive else 0)

            for v_vec, m_vec in zip(values, masks):
                for v, m in zip(v_vec, m_vec):
                    if m > 0:
                        all_values.append(v)

        if normalization_stats is None:
            all_values = np.array(all_values)
            self.mean = np.mean(all_values) if len(all_values) > 0 else 0.0
            self.std = np.std(all_values) if len(all_values) > 0 else 1.0
        else:
            self.mean = normalization_stats['mean']
            self.std = normalization_stats['std']

        for i in range(len(self.data)):
            norm_values = []
            for v_vec, m_vec in zip(self.data[i]['values'], self.data[i]['masks']):
                norm = [(v - self.mean)/self.std if m>0 else 0.0
                        for v, m in zip(v_vec, m_vec)]
                norm_values.append(norm)

            self.data[i] = {
                'times': torch.tensor(self.data[i]['times'], dtype=torch.float32),
                'values': torch.tensor(norm_values, dtype=torch.float32),
                'masks': torch.tensor(self.data[i]['masks'], dtype=torch.float32)
            }

    def get_normalization_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.static_data[idx]

def hybrid_collate_fn(batch):
    data_list, label_list, static_list = zip(*batch)
    lengths = [len(d['times']) for d in data_list]
    max_len = max(lengths)
    feat_dim = data_list[0]['values'].shape[-1]
    batch_size = len(data_list)

    padded_times = torch.zeros(batch_size, max_len)
    padded_values = torch.zeros(batch_size, max_len, feat_dim)
    padded_masks = torch.zeros(batch_size, max_len, feat_dim)

    for i, d in enumerate(data_list):
        l = lengths[i]
        padded_times[i, :l] = d['times']
        padded_values[i, :l] = d['values']
        padded_masks[i, :l] = d['masks']

    temporal_batch = {
        'times': padded_times,
        'values': padded_values,
        'masks': padded_masks,
        'lengths': torch.tensor(lengths)
    }
    return temporal_batch, torch.tensor(label_list, dtype=torch.float32), torch.stack(static_list)

# ==============================================================================
# 3. RNN Policy Network
# ==============================================================================

class RNNPolicyNetwork(nn.Module):
    """RNN with Gaussian policy"""
    def __init__(self, input_dim, hidden_dim, latent_dim, time_dim=32):
        super().__init__()
        self.rnn_cell = TimeEmbeddedRNNCell(input_dim, hidden_dim, time_dim)

        # Output mean and log_std for Gaussian policy
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logstd = nn.Linear(hidden_dim, latent_dim)

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

    def forward(self, batch_data, deterministic=False, temperature=1.0):
        times = batch_data['times'].to(DEVICE)
        values = batch_data['values'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)
        lengths = batch_data['lengths'].to(DEVICE)

        h = self.rnn_cell(times, values, masks, lengths)

        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std) * temperature

        policy_dist = dist.Normal(mean, std)

        if deterministic:
            z = mean
            log_prob = None
        else:
            z = policy_dist.rsample()
            log_prob = policy_dist.log_prob(z).sum(dim=-1)

        return z, log_prob, mean

class SupervisedHead(nn.Module):
    """Enhanced supervised head with residual connection"""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))

# ==============================================================================
# 4. Enhanced Pretraining
# ==============================================================================

def pretrain_rnn_enhanced(policy_net, train_loader, val_loader, epochs=50):
    """Enhanced pretraining with better architecture and longer training"""
    print("  [Enhanced Pretraining] Longer supervised learning...")

    supervised_head = SupervisedHead(
        policy_net.latent_dim + len(FIXED_FEATURES)
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(supervised_head.parameters()),
        lr=0.001
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    criterion = nn.BCELoss()

    best_auc = 0
    best_state = None
    patience = 12
    counter = 0

    for epoch in range(epochs):
        policy_net.train()
        supervised_head.train()

        for t_data, labels, s_data in train_loader:
            labels = labels.to(DEVICE)
            s_data = s_data.to(DEVICE)

            z, _, _ = policy_net(t_data, deterministic=True)
            combined = torch.cat([z, s_data], dim=1)

            preds = supervised_head(combined).squeeze(-1)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy_net.parameters()) + list(supervised_head.parameters()),
                max_norm=1.0
            )
            optimizer.step()

        # Validation
        if (epoch + 1) % 3 == 0:
            policy_net.eval()
            supervised_head.eval()
            all_preds, all_labels = [], []

            with torch.no_grad():
                for t_data, labels, s_data in val_loader:
                    s_data = s_data.to(DEVICE)
                    z, _, _ = policy_net(t_data, deterministic=True)
                    combined = torch.cat([z, s_data], dim=1)
                    preds = supervised_head(combined).squeeze(-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

            val_aupr = average_precision_score(all_labels, all_preds)
            scheduler.step(val_aupr)

            print(f"    Pretrain Epoch {epoch+1} | Val AUPR: {val_aupr:.4f}")

            if val_aupr > best_auc:
                best_auc = val_aupr
                best_state = copy.deepcopy(policy_net.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

    if best_state is not None:
        policy_net.load_state_dict(best_state)

    print(f"  [Enhanced Pretraining] Completed. Best Val AUPR: {best_auc:.4f}")
    return policy_net

# ==============================================================================
# 5. Feature Extraction with Enriched Temporal Features
# ==============================================================================

def extract_enriched_features_and_logprobs(policy_net, loader, deterministic=False, temperature=1.0):
    """
    Extract enriched features: [Static + Last + Mean + Std + Z]
    Adds more temporal statistics
    """
    policy_net.eval() if deterministic else policy_net.train()

    all_features = []
    all_labels = []
    all_log_probs = []

    with torch.set_grad_enabled(not deterministic):
        for t_data, labels, s_data in loader:
            z, log_prob, mean = policy_net(t_data, deterministic=deterministic, temperature=temperature)
            z_np = (mean if deterministic else z).detach().cpu().numpy()

            vals = t_data['values'].cpu().numpy()
            masks = t_data['masks'].cpu().numpy()

            batch_last_vals = []
            batch_mean_vals = []
            batch_std_vals = []

            for i in range(len(vals)):
                patient_last = []
                patient_mean = []
                patient_std = []

                for f_idx in range(vals.shape[2]):
                    f_vals = vals[i, :, f_idx]
                    f_mask = masks[i, :, f_idx]
                    valid_idx = np.where(f_mask > 0)[0]

                    if len(valid_idx) > 0:
                        valid_vals = f_vals[valid_idx]
                        patient_last.append(valid_vals[-1])
                        patient_mean.append(np.mean(valid_vals))
                        patient_std.append(np.std(valid_vals) if len(valid_vals) > 1 else 0.0)
                    else:
                        patient_last.append(0.0)
                        patient_mean.append(0.0)
                        patient_std.append(0.0)

                batch_last_vals.append(patient_last)
                batch_mean_vals.append(patient_mean)
                batch_std_vals.append(patient_std)

            last_vals_arr = np.array(batch_last_vals)
            mean_vals_arr = np.array(batch_mean_vals)
            std_vals_arr = np.array(batch_std_vals)
            s_np = s_data.numpy()

            # Enriched concatenation: [Static + Last + Mean + Std + Z]
            combined = np.hstack([s_np, last_vals_arr, mean_vals_arr, std_vals_arr, z_np])

            all_features.append(combined)
            all_labels.extend(labels.numpy())

            if not deterministic and log_prob is not None:
                all_log_probs.append(log_prob)

    features = np.vstack(all_features)
    labels = np.array(all_labels)
    log_probs = torch.cat(all_log_probs) if all_log_probs else None

    return features, labels, log_probs

# ==============================================================================
# 6. Conservative RL Training
# ==============================================================================

def train_policy_conservative_rl(
    policy_net,
    train_loader,
    val_loader,
    tabpfn_params,
    epochs=80,
    update_tabpfn_every=5
):
    """
    Conservative RL: smaller updates, lower temperature, AUPR-focused
    """

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.0001)  # Lower LR

    best_val_auc = 0
    best_state = None
    patience = 25
    patience_counter = 0

    print("  [Conservative RL] AUPR-focused training with lower exploration...")

    tabpfn_model = None

    for epoch in range(epochs):
        # Very conservative temperature annealing
        temperature = max(0.3, 0.8 - epoch / (epochs * 0.7))

        policy_net.train()

        # Sample features
        X_train, y_train, log_probs_train = extract_enriched_features_and_logprobs(
            policy_net, train_loader, deterministic=False, temperature=temperature
        )

        # Train TabPFN
        if epoch % update_tabpfn_every == 0 or tabpfn_model is None:
            tabpfn_model = TabPFNClassifier(**tabpfn_params)
            tabpfn_model.fit(X_train, y_train)

        # (Removed the per-epoch stochastic val forward pass: val_aupr was only
        # feeding the dropped reward term. Val is now evaluated solely in the
        # deterministic validation block below, every 5 epochs.)

        # Per-sample probability quality (computed on TRAIN samples — these are
        # the samples whose log_probs we update against, so the reward must be
        # per-sample and train-derived).
        y_train_proba = tabpfn_model.predict_proba(X_train)[:, 1]
        rewards_smooth = np.where(y_train == 1, y_train_proba, 1 - y_train_proba)

        # Reward is the per-sample train quality only. The previous
        # `+ 0.5 * val_aupr` term mixed a val-set scalar into the training
        # signal (val leakage into gradients) while being nearly cancelled by
        # the mean-subtraction below anyway. Dropped on both counts; val now
        # serves a single role: early-stopping / checkpoint selection.
        rewards_combined = rewards_smooth

        # Normalize
        rewards_tensor = torch.tensor(rewards_combined, dtype=torch.float32).to(DEVICE)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # Policy update
        log_probs_train = log_probs_train.to(DEVICE)
        policy_loss = -(log_probs_train * rewards_tensor).mean()

        # Minimal entropy (trust the good initialization)
        entropy_bonus = 0.001 * log_probs_train.mean()

        total_loss = policy_loss - entropy_bonus

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)  # Smaller gradients
        optimizer.step()

        # Validation
        if (epoch + 1) % 5 == 0:
            policy_net.eval()
            with torch.no_grad():
                X_val_det, y_val_det, _ = extract_enriched_features_and_logprobs(
                    policy_net, val_loader, deterministic=True
                )

                X_train_det, y_train_det, _ = extract_enriched_features_and_logprobs(
                    policy_net, train_loader, deterministic=True
                )

                tabpfn_val_model = TabPFNClassifier(**tabpfn_params)
                tabpfn_val_model.fit(X_train_det, y_train_det)

                y_val_proba_det = tabpfn_val_model.predict_proba(X_val_det)[:, 1]

                val_auc = roc_auc_score(y_val_det, y_val_proba_det)
                val_aupr_det = average_precision_score(y_val_det, y_val_proba_det)

                print(f"    Epoch {epoch+1:3d} | Temp: {temperature:.3f} | "
                      f"Val AUC: {val_auc:.4f} | Val AUPR: {val_aupr_det:.4f}")

                # Early stopping on AUPR
                if val_aupr_det > best_val_auc:
                    best_val_auc = val_aupr_det
                    best_state = copy.deepcopy(policy_net.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break

    if best_state is not None:
        policy_net.load_state_dict(best_state)

    return policy_net

# ==============================================================================
# 7. Main Execution
# ==============================================================================

def main():
    print("="*80)
    print("RL POLICY V3 (Enhanced) vs BASELINE")
    print("="*80)

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)

    print("Encoding static features...")
    encoder = SimpleStaticEncoder(FIXED_FEATURES)
    encoder.fit(patients.patientList)

    print(f"Input: {len(temporal_feats)} Temporal + {len(FIXED_FEATURES)} Static Features")

    metrics_rl = {k: [] for k in ['auc', 'auc_pr']}
    metrics_baseline = {k: [] for k in ['auc', 'auc_pr']}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, seed=xseed)):
        print(f"\n{'='*80}")
        print(f"Fold {fold}")
        print('='*80)

        train_p_obj, val_p_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42+fold)
        train_p = train_p_obj.patientList
        val_p = val_p_obj.patientList
        test_p_list = test_p.patientList

        train_ds = HybridDataset(train_p, temporal_feats, encoder)
        stats = train_ds.get_normalization_stats()
        val_ds = HybridDataset(val_p, temporal_feats, encoder, stats)
        test_ds = HybridDataset(test_p_list, temporal_feats, encoder, stats)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=hybrid_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)

        # Larger capacity
        latent_dim = 28
        policy_net = RNNPolicyNetwork(
            input_dim=len(temporal_feats),
            hidden_dim=20,
            latent_dim=latent_dim,
            time_dim=32
        ).to(DEVICE)

        tabpfn_params = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

        # STEP 1: Enhanced Pretraining (longer)
        policy_net = pretrain_rnn_enhanced(policy_net, train_loader, val_loader, epochs=50)

        # STEP 2: Conservative RL Fine-tuning
        policy_net = train_policy_conservative_rl(
            policy_net,
            train_loader,
            val_loader,
            tabpfn_params,
            epochs=80,
            update_tabpfn_every=5
        )

        # Final Evaluation
        print("\n  [Final Test Evaluation]")
        policy_net.eval()

        with torch.no_grad():
            X_train_final, y_train_final, _ = extract_enriched_features_and_logprobs(
                policy_net, train_loader, deterministic=True
            )
            X_test_final, y_test_final, _ = extract_enriched_features_and_logprobs(
                policy_net, test_loader, deterministic=True
            )

        final_tabpfn = TabPFNClassifier(**tabpfn_params)
        final_tabpfn.fit(X_train_final, y_train_final)

        y_test_proba = final_tabpfn.predict_proba(X_test_final)[:, 1]
        y_test_pred = (y_test_proba > 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test_final, y_test_pred).ravel()
        prec, rec, _ = precision_recall_curve(y_test_final, y_test_proba)

        fold_auc = roc_auc_score(y_test_final, y_test_proba)
        fold_aupr = auc(rec, prec)

        metrics_rl['auc'].append(fold_auc)
        metrics_rl['auc_pr'].append(fold_aupr)

        fpr, tpr, _ = roc_curve(y_test_final, y_test_proba)
        ax1.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC = {fold_auc:.3f})")

        print(f"  RL Test AUC: {fold_auc:.4f} | Test AUPR: {fold_aupr:.4f}")

        # BASELINE
        print("\n  [Baseline] Standard TabPFN (Last + Static)...")

        df_train_temp = train_p_obj.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])
        df_test_temp = test_p.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last", getUntilAkiPositive=True
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])

        df_train_enc, df_test_enc, _ = encodeCategoricalData(df_train_temp, df_test_temp)

        X_tr_b = df_train_enc.drop(columns=["akd"]).fillna(0)
        y_tr_b = df_train_enc["akd"]
        X_te_b = df_test_enc.drop(columns=["akd"]).fillna(0)
        y_te_b = df_test_enc["akd"]

        tabpfn_base = TabPFNClassifier(**tabpfn_params)
        tabpfn_base.fit(X_tr_b, y_tr_b)

        y_prob_b = tabpfn_base.predict_proba(X_te_b)[:, 1]
        prec_b, rec_b, _ = precision_recall_curve(y_te_b, y_prob_b)

        baseline_auc = roc_auc_score(y_te_b, y_prob_b)
        baseline_aupr = auc(rec_b, prec_b)

        metrics_baseline['auc'].append(baseline_auc)
        metrics_baseline['auc_pr'].append(baseline_aupr)

        fpr_b, tpr_b, _ = roc_curve(y_te_b, y_prob_b)
        ax2.plot(fpr_b, tpr_b, lw=2, label=f"Fold {fold} (AUC = {baseline_auc:.3f})")

        print(f"  Baseline Test AUC: {baseline_auc:.4f} | Test AUPR: {baseline_aupr:.4f}")
        print(f"  Fold {fold} Results -> RL: {fold_auc:.3f} vs Baseline: {baseline_auc:.3f}")

    # Final Plot
    # for ax in [ax1, ax2]:
    #     ax.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2)
    #     ax.set_xlim([0.0, 1.0])
    #     ax.set_ylim([0.0, 1.05])
    #     ax.set_xlabel("False Positive Rate")
    #     ax.set_ylabel("True Positive Rate")
    #     ax.legend(loc="lower right")

    # ax1.set_title("RL Policy V3 + TabPFN Judge")
    # ax2.set_title("Baseline (Last + Static)")
    #plt.tight_layout()
    #plt.savefig("result/tab_rlv3_vs_baseline.png", dpi=300)
    #print("\nPlot saved to result/tab_rlv3_vs_baseline.png")

    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    def print_stat(name, rl_metrics, base_metrics=None):
        rl_mean, rl_std = np.mean(rl_metrics), np.std(rl_metrics)
        if base_metrics is not None:
            base_mean, base_std = np.mean(base_metrics), np.std(base_metrics)
            improvement = ((rl_mean - base_mean) / base_mean) * 100
            symbol = "✓" if rl_mean > base_mean else "✗"
            print(f"{name:15s} | RL: {rl_mean:.4f} ± {rl_std:.4f}  vs  Baseline: {base_mean:.4f} ± {base_std:.4f}  ({improvement:+.2f}%) {symbol}")
        else:
            print(f"{name:15s} | RL: {rl_mean:.4f} ± {rl_std:.4f}) {symbol}")

    print_stat("AUC", metrics_rl['auc'])
    print_stat("AUC-PR", metrics_rl['auc_pr'])

if __name__ == "__main__":
    main()