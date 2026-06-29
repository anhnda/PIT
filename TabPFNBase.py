"""
TabPFN BASELINE: [Last Value + Static Context] -> TabPFN

This uses the same feature set as the baseline in TBoostv2.py but replaces
XGBoost with TabPFN (Tabular Pre-trained Feature Network) as the classifier.

Features:
- Last Values from temporal features
- Static Context features
- TabPFN Classifier with ensemble configurations
"""

import pandas as pd
import numpy as np
import sys
import os
from matplotlib import pyplot as plt
import random

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

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

xseed = 42
seed_everything()
PT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PT)

from utils.class_patient import Patients
from utils.prepare_data import trainTestPatients, encodeCategoricalData

# Import TabPFN
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("Warning: tabpfn not installed. Please install with: pip install tabpfn")

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    if not TABPFN_AVAILABLE:
        print("Error: TabPFN is not available. Please install it first.")
        return

    print("="*80)
    print("TabPFN BASELINE: [Last Value + Static Context] -> TabPFN")
    print("="*80)

    # Load patients
    print("Loading patients...")
    patients = Patients.loadPatients()
    _info = patients.applyLandmarkHorizon(
        landmark=pd.Timedelta(hours=24), horizon=pd.Timedelta(hours=48)
    )
    print(f"[Landmark] cohort {_info['n_before']} -> {_info['n_after']} | "
          f"excluded {_info['n_excluded_pre_landmark']} pre-landmark | "
          f"pos_rate {_info['pos_rate']:.3f}")

    # Store metrics
    metrics = {k: [] for k in ['auc', 'acc', 'spec', 'prec', 'rec', 'auc_pr']}

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients, seed=xseed)):
        print(f"\n--- Fold {fold} ---")

        # Extract features: Last Values + Static
        print("  Extracting baseline features (Last + Static)...")

        df_train = train_full.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last",
            getUntilAkiPositive=False
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])

        df_test = test_p.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last",
            getUntilAkiPositive=False
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])

        # Encode categorical features
        df_train_enc, df_test_enc, _ = encodeCategoricalData(df_train, df_test)

        # Prepare training and test sets
        X_train = df_train_enc.drop(columns=["akd"]).fillna(0)
        y_train = df_train_enc["akd"]
        X_test = df_test_enc.drop(columns=["akd"]).fillna(0)
        y_test = df_test_enc["akd"]

        print(f"    Training samples: {len(X_train)}, Features: {X_train.shape[1]}")
        print(f"    Test samples: {len(X_test)}")

        # Train TabPFN classifier
        print("  Training TabPFN classifier...")

        # Configure TabPFN
        # Note: TabPFN has limitations on dataset size (max 1024 samples, 100 features)
        # We may need to subsample if the dataset is too large
        max_train_samples = min(1024, len(X_train))
        if len(X_train) > max_train_samples:
            print(f"    Subsampling training data to {max_train_samples} samples (TabPFN limitation)")
            # Stratified sampling to maintain class balance
            pos_indices = np.where(y_train == 1)[0]
            neg_indices = np.where(y_train == 0)[0]

            pos_ratio = len(pos_indices) / len(y_train)
            n_pos = int(max_train_samples * pos_ratio)
            n_neg = max_train_samples - n_pos

            sampled_pos = np.random.choice(pos_indices, size=min(n_pos, len(pos_indices)), replace=False)
            sampled_neg = np.random.choice(neg_indices, size=min(n_neg, len(neg_indices)), replace=False)
            sampled_indices = np.concatenate([sampled_pos, sampled_neg])
            np.random.shuffle(sampled_indices)

            X_train_sample = X_train.iloc[sampled_indices]
            y_train_sample = y_train.iloc[sampled_indices]
        else:
            X_train_sample = X_train
            y_train_sample = y_train

        # Check feature count
        if X_train.shape[1] > 100:
            print(f"    Warning: {X_train.shape[1]} features exceeds TabPFN's recommended limit of 100")
            print(f"    TabPFN may not perform optimally with this many features")

        try:
            tabpfn = TabPFNClassifier(
                device='cuda',  # Use 'cuda' if GPU is available
            )

            tabpfn.fit(X_train_sample, y_train_sample)

            # Predictions
            y_prob = tabpfn.predict_proba(X_test)[:, 1]
            y_pred = (y_prob > 0.5).astype(int)

            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            prec, rec, _ = precision_recall_curve(y_test, y_prob)

            metrics['auc'].append(roc_auc_score(y_test, y_prob))
            metrics['acc'].append(accuracy_score(y_test, y_pred))
            metrics['spec'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            metrics['prec'].append(precision_score(y_test, y_pred, zero_division=0))
            metrics['rec'].append(recall_score(y_test, y_pred, zero_division=0))
            metrics['auc_pr'].append(auc(rec, prec))

            # Plot ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            ax.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC = {metrics['auc'][-1]:.3f})")

            print(f"  Fold {fold} Results -> AUC: {metrics['auc'][-1]:.4f}, "
                  f"AUC-PR: {metrics['auc_pr'][-1]:.4f}")

        except Exception as e:
            print(f"    Error in fold {fold}: {str(e)}")
            print(f"    Skipping fold {fold}")
            continue

    # Final Plot Configuration
    ax.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("TabPFN Baseline: Last Value + Static Context", fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("result/tabpfn_baseline.png", dpi=300)
    print("\nPlot saved to result/tabpfn_baseline.png")

    # Final Stats
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    if len(metrics['auc']) > 0:
        def print_stat(name, metric_list):
            mean_val = np.mean(metric_list)
            std_val = np.std(metric_list)
            print(f"{name:15s} | {mean_val:.4f} ± {std_val:.4f}")

        print_stat("AUC", metrics['auc'])
        print_stat("AUC-PR", metrics['auc_pr'])
        print_stat("Accuracy", metrics['acc'])
        print_stat("Specificity", metrics['spec'])
        print_stat("Precision", metrics['prec'])
        print_stat("Recall", metrics['rec'])
    else:
        print("No successful folds to report.")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
