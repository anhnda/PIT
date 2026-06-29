"""
CatBoost BASELINE: [Last Value + Static Context] -> CatBoost

Simple baseline without any reinforcement learning or RNN components.
Just extracts last values and static features, then trains CatBoost.

This serves as the baseline to compare against more complex methods.
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
    average_precision_score,
    auc,
)

from catboost import CatBoostClassifier

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

xseed = 42
seed_everything(xseed)

PT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PT)

from constants import NULLABLE_MEASURES
from utils.class_patient import Patients
from utils.prepare_data import trainTestPatients, encodeCategoricalData

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    print("="*80)
    print("CatBoost BASELINE: [Last Value + Static Context]")
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

    print(f"Total patients: {len(patients)}")
    print(f"Positive cases: {sum([p.akdPositive for p in patients.patientList])}")
    print(f"Negative cases: {sum([not p.akdPositive for p in patients.patientList])}")

    # Store metrics
    metrics = {k: [] for k in ['auc', 'acc', 'spec', 'prec', 'rec', 'auc_pr']}

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for fold, (train_full, test_p) in enumerate(trainTestPatients(patients,seed=xseed)):
        print(f"\n{'='*80}")
        print(f"Fold {fold}")
        print('='*80)

        # Extract "Last Values + Static" using getMeasuresBetween
        print("  Extracting baseline features (Last + Static)...")

        df_train = train_full.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last",
            getUntilAkiPositive=False
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])

        df_test = test_p.getMeasuresBetween(
            pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last",
            getUntilAkiPositive=False
        ).drop(columns=["subject_id", "hadm_id", "stay_id"])

        # Encode categorical data
        df_train_enc, df_test_enc, _ = encodeCategoricalData(df_train, df_test)

        # Prepare training and test sets
        X_train = df_train_enc.drop(columns=["akd"]).fillna(0)
        y_train = df_train_enc["akd"]
        X_test = df_test_enc.drop(columns=["akd"]).fillna(0)
        y_test = df_test_enc["akd"]

        print(f"    Training samples: {len(X_train)}, Features: {X_train.shape[1]}")
        print(f"    Test samples: {len(X_test)}")
        print(f"    Train positive rate: {y_train.mean():.3f}")
        print(f"    Test positive rate: {y_test.mean():.3f}")

        # Calculate class imbalance ratio
        ratio = np.sum(y_train == 0) / (np.sum(y_train == 1) + 1e-6)
        print(f"    Class imbalance ratio: {ratio:.2f}")

        # Train CatBoost
        print("  Training CatBoost classifier...")

        catboost = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function='Logloss',
            eval_metric='AUC',
            scale_pos_weight=ratio,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            task_type='CPU'  # Use 'GPU' if available
        )

        catboost.fit(X_train, y_train)

        # Predictions
        y_prob = catboost.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        prec, rec, _ = precision_recall_curve(y_test, y_prob)

        fold_auc = roc_auc_score(y_test, y_prob)
        fold_acc = accuracy_score(y_test, y_pred)
        fold_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        fold_prec = precision_score(y_test, y_pred, zero_division=0)
        fold_rec = recall_score(y_test, y_pred, zero_division=0)
        fold_aupr = auc(rec, prec)

        metrics['auc'].append(fold_auc)
        metrics['acc'].append(fold_acc)
        metrics['spec'].append(fold_spec)
        metrics['prec'].append(fold_prec)
        metrics['rec'].append(fold_rec)
        metrics['auc_pr'].append(fold_aupr)

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC = {fold_auc:.3f})")

        print(f"  Fold {fold} Results:")
        print(f"    AUC: {fold_auc:.4f}")
        print(f"    AUC-PR: {fold_aupr:.4f}")
        print(f"    Accuracy: {fold_acc:.4f}")
        print(f"    Sensitivity (Recall): {fold_rec:.4f}")
        print(f"    Specificity: {fold_spec:.4f}")
        print(f"    Precision: {fold_prec:.4f}")

    # Final Plot Configuration
    ax.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("CatBoost Baseline: Last Value + Static Context", fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Create result directory if it doesn't exist
    os.makedirs("result", exist_ok=True)
    plt.savefig("result/catboost_baseline.png", dpi=300)
    print("\nPlot saved to result/catboost_baseline.png")

    # Final Stats
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    def print_stat(name, metric_list):
        mean_val = np.mean(metric_list)
        std_val = np.std(metric_list)
        print(f"{name:20s} | {mean_val:.4f} ± {std_val:.4f}")

    print_stat("AUC", metrics['auc'])
    print_stat("AUC-PR", metrics['auc_pr'])
    print_stat("Accuracy", metrics['acc'])
    print_stat("Sensitivity (Recall)", metrics['rec'])
    print_stat("Specificity", metrics['spec'])
    print_stat("Precision", metrics['prec'])

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
