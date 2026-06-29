"""
Time-Embedded RNN for AKI prediction using temporal features.

Key Innovation: Instead of ODE solver, use time embedding:
1. Embed time gaps into rich representation using MLP
2. Concatenate [Values, Masks, Time_Embedding] as GRU input
3. No ODE solver needed → Faster and simpler than ODE-RNN
4. Still captures irregular temporal patterns

Advantages over ODE-RNN:
- ✓ Simpler architecture (no ODE solver)
- ✓ Faster training (no numerical integration)
- ✓ More stable (no ODE numerical issues)
- ✓ Still handles irregular time gaps
"""

import os
import pandas as pd
import numpy as np
import sys
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
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

PT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PT)

from constants import NULLABLE_MEASURES
from utils.class_patient import Patients
from utils.prepare_data import trainTestPatients

# Fixed/demographic features to exclude (same as MLP_OnlyTime.py)
FIXED_FEATURES = [
    "age", "gender", "race",
    "chronic_pulmonary_disease", "ckd_stage", "congestive_heart_failure",
    "dka_type", "history_aci", "history_ami", "hypertension",
    "liver_disease", "macroangiopathy", "malignant_cancer",
    "microangiopathy", "uti",
    "oasis", "saps2", "sofa",
    "mechanical_ventilation", "use_NaHCO3",
    "preiculos", "gcs_unable",
    "egfr",  # Data leak
]

# Configuration
LABEL_COLUMN = "akd"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")


# ============================================================================
# 1. DATA PREPARATION (Same as ODETime.py)
# ============================================================================

def get_all_temporal_features(patients):
    """Get the complete set of temporal features across all patients."""
    id_columns = ["subject_id", "hadm_id", "stay_id"]
    label_column = "akd"

    df = patients.getMeasuresBetween(
        pd.Timedelta(hours=-6),
        pd.Timedelta(hours=24),
        "last",
        getUntilAkiPositive=False,
    )

    all_features = [col for col in df.columns
                   if col not in id_columns + [label_column]
                   and col not in FIXED_FEATURES]

    return sorted(all_features)


def extract_temporal_data(patient, feature_names, time_window_start=-6, time_window_end=24):
    """Extract temporal measurements over a FIXED observation window.

    The window is the same for every patient and never depends on outcome.
    After applyLandmarkHorizon, all retained patients have akdTime > landmark,
    so the old onset-based truncation (which previously cut the window at
    aki_cutoff_hours for positives) is both unnecessary and a source of
    leakage; it has been removed.
    """
    intime = patient.intime

    # Fixed window end — independent of the label (no onset-based cutoff).
    effective_end = time_window_end

    # Get temporal measures
    temporal_measures = {}
    for measure_name in feature_names:
        if measure_name in patient.measures:
            measure_values = patient.measures[measure_name]

            if hasattr(measure_values, 'keys') and hasattr(measure_values, 'values'):
                filtered_dict = {}
                for timestamp, value in measure_values.items():
                    ts = pd.Timestamp(timestamp)
                    hours_from_admission = (ts - intime).total_seconds() / 3600

                    if time_window_start <= hours_from_admission <= effective_end:
                        filtered_dict[ts] = value

                temporal_measures[measure_name] = filtered_dict
            else:
                temporal_measures[measure_name] = {}
        else:
            temporal_measures[measure_name] = {}

    # Get all unique timestamps
    all_timestamps = set()
    for measure_dict in temporal_measures.values():
        for timestamp in measure_dict.keys():
            all_timestamps.add(timestamp)

    if not all_timestamps:
        return None, None, None

    all_timestamps = sorted(all_timestamps)

    # Convert to hours from admission (keep absolute hours, don't normalize yet)
    times_hours = [(t - intime).total_seconds() / 3600 for t in all_timestamps]

    # Build values and masks
    values = []
    masks = []

    for timestamp in all_timestamps:
        value_vec = []
        mask_vec = []

        for feature_name in feature_names:
            measure_dict = temporal_measures[feature_name]

            if timestamp in measure_dict:
                value_vec.append(float(measure_dict[timestamp]))
                mask_vec.append(1.0)
            else:
                value_vec.append(0.0)
                mask_vec.append(0.0)

        values.append(value_vec)
        masks.append(mask_vec)

    return times_hours, values, masks


class IrregularTimeSeriesDataset(Dataset):
    """Dataset for irregular time series data with normalization."""

    def __init__(self, patients, feature_names, normalization_stats=None):
        self.data = []
        self.labels = []
        self.feature_names = feature_names

        # Collect all values for normalization
        all_values = []

        for patient in patients.patientList:
            times, values, masks = extract_temporal_data(patient, feature_names)

            if times is None:
                continue

            self.data.append({
                'times': times,
                'values': values,
                'masks': masks,
            })
            self.labels.append(1 if patient.akdPositive else 0)

            # Collect observed values for normalization
            for value_vec, mask_vec in zip(values, masks):
                for val, mask in zip(value_vec, mask_vec):
                    if mask > 0:
                        all_values.append(val)

        # Compute or use provided normalization statistics
        if normalization_stats is None:
            all_values = np.array(all_values)
            self.mean = np.mean(all_values) if len(all_values) > 0 else 0.0
            self.std = np.std(all_values) if len(all_values) > 0 else 1.0
            if self.std == 0:
                self.std = 1.0
        else:
            self.mean = normalization_stats['mean']
            self.std = normalization_stats['std']

        # Normalize the data
        for i in range(len(self.data)):
            values = self.data[i]['values']
            masks = self.data[i]['masks']

            # Normalize only observed values
            normalized_values = []
            for value_vec, mask_vec in zip(values, masks):
                norm_vec = [(v - self.mean) / self.std if m > 0 else 0.0
                           for v, m in zip(value_vec, mask_vec)]
                normalized_values.append(norm_vec)

            # Convert to tensors
            self.data[i] = {
                'times': torch.tensor(self.data[i]['times'], dtype=torch.float32),
                'values': torch.tensor(normalized_values, dtype=torch.float32),
                'masks': torch.tensor(self.data[i]['masks'], dtype=torch.float32),
            }

    def get_normalization_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def collate_fn(batch):
    """Custom collate function with padding for batched processing."""
    data, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.float32)

    lengths = [len(d['times']) for d in data]
    max_len = max(lengths)

    feature_dim = data[0]['values'].shape[-1]
    batch_size = len(data)

    padded_times = torch.zeros(batch_size, max_len)
    padded_values = torch.zeros(batch_size, max_len, feature_dim)
    padded_masks = torch.zeros(batch_size, max_len, feature_dim)

    for i, d in enumerate(data):
        seq_len = lengths[i]
        padded_times[i, :seq_len] = d['times']
        padded_values[i, :seq_len] = d['values']
        padded_masks[i, :seq_len] = d['masks']

    batch_data = {
        'times': padded_times,
        'values': padded_values,
        'masks': padded_masks,
        'lengths': torch.tensor(lengths, dtype=torch.long)
    }

    return batch_data, labels


# ============================================================================
# 2. TIME-EMBEDDED RNN MODEL
# ============================================================================

class TimeEmbeddedRNNCell(nn.Module):
    """Time-Embedded RNN cell with time gap embedding."""

    def __init__(self, input_dim, hidden_dim, time_dim=32):
        super(TimeEmbeddedRNNCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim

        # Time embedding network: 1D time gap → time_dim dimensional embedding
        self.time_embedder = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),  # Smooth activation for gradients
            nn.Linear(time_dim, time_dim)
        )

        # Learned initial hidden state
        self.h0 = nn.Parameter(torch.randn(hidden_dim) * 0.1)

        # GRU cell takes [values (input_dim) + masks (input_dim) + time_embedding (time_dim)]
        self.gru_cell = nn.GRUCell(input_dim * 2 + time_dim, hidden_dim)

    def forward(self, batch_times, batch_values, batch_masks, lengths):
        """
        Args:
            batch_times: [batch_size, max_seq_len] timestamps in hours
            batch_values: [batch_size, max_seq_len, input_dim]
            batch_masks: [batch_size, max_seq_len, input_dim]
            lengths: [batch_size] actual sequence lengths
        Returns:
            h: [batch_size, hidden_dim] final hidden states
        """
        batch_size = batch_times.size(0)
        max_seq_len = batch_times.size(1)

        # Initialize hidden state for entire batch
        h = self.h0.unsqueeze(0).repeat(batch_size, 1)  # [batch_size, hidden_dim]

        # Process each time step
        for i in range(max_seq_len):
            # Calculate time delta from previous step
            if i > 0:
                delta_t = batch_times[:, i] - batch_times[:, i-1]  # [batch_size]
            else:
                delta_t = torch.zeros(batch_size, device=h.device)

            # Embed the time gap
            # [batch_size, 1] → [batch_size, time_dim]
            time_embedding = self.time_embedder(delta_t.unsqueeze(-1))

            # Combine: [values, masks, time_embedding]
            combined_input = torch.cat([
                batch_values[:, i],     # [batch_size, input_dim]
                batch_masks[:, i],      # [batch_size, input_dim]
                time_embedding          # [batch_size, time_dim]
            ], dim=-1)  # [batch_size, input_dim*2 + time_dim]

            # Update hidden state with GRU
            h_new = self.gru_cell(combined_input, h)  # [batch_size, hidden_dim]

            # Masking: Only update patients that haven't exceeded their sequence length
            valid_mask = (i < lengths).float().unsqueeze(-1)  # [batch_size, 1]
            h = valid_mask * h_new + (1 - valid_mask) * h

        return h


class TimeEmbeddedRNNModel(nn.Module):
    """Complete Time-Embedded RNN model for classification."""

    def __init__(self, input_dim, hidden_dim, time_dim=32, output_dim=1):
        super(TimeEmbeddedRNNModel, self).__init__()
        self.rnn_cell = TimeEmbeddedRNNCell(input_dim, hidden_dim, time_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

    def forward(self, batch_data):
        """
        Args:
            batch_data: Dict with 'times', 'values', 'masks', 'lengths'
        Returns:
            predictions: [batch_size] probability of positive class
        """
        times = batch_data['times'].to(DEVICE)
        values = batch_data['values'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)
        lengths = batch_data['lengths'].to(DEVICE)

        # Get hidden states for entire batch
        hidden_states = self.rnn_cell(times, values, masks, lengths)

        # Classify
        predictions = self.classifier(hidden_states).squeeze(-1)

        return predictions


# ============================================================================
# 3. TRAINING AND EVALUATION
# ============================================================================

def load_and_prepare_patients(
    landmark=pd.Timedelta(hours=24),
    horizon=pd.Timedelta(hours=48),
):
    """Load patients, remove missing data, and apply the landmark protocol.

    The landmark/horizon step (applyLandmarkHorizon) converts the cohort into a
    proper future-prediction task and removes the outcome-dependent windowing
    leak (Reviewer 6, point 2). After this call, every downstream feature
    extraction MUST use a fixed window [..., landmark] and MUST NOT pass
    getUntilAkiPositive=True.
    """
    patients = Patients.loadPatients()
    print(f"Loaded {len(patients)} patients")

    patients.fillMissingMeasureValue(NULLABLE_MEASURES, 0)

    measures = patients.getMeasures()
    for measure, count in measures.items():
        if count < len(patients) * 80 / 100:
            patients.removeMeasures([measure])

    patients.removePatientByMissingFeatures()
    print(f"After cleanup: {len(patients)} patients")

    # --- Landmark prediction protocol (removes label leakage) ---------------
    info = patients.applyLandmarkHorizon(landmark=landmark, horizon=horizon)
    print(
        f"[Landmark] L={info['landmark_hours']:.0f}h H={info['horizon_hours']:.0f}h | "
        f"excluded {info['n_excluded_pre_landmark']} pre-landmark AKI | "
        f"cohort {info['n_before']} -> {info['n_after']} | "
        f"pos {info['n_positive']} ({info['pos_rate']:.2%})"
    )

    aki_count = sum([1 for p in patients if p.akdPositive])
    print(f"AKI positive (horizon): {aki_count} ({aki_count / len(patients):.2%})")

    return patients


def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    """Train the Time-Embedded RNN model."""
    model.train()

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        total_loss = 0
        num_batches = 0

        for batch_data, labels in train_loader:
            labels = labels.to(DEVICE)

            # Forward pass
            predictions = model(batch_data)
            loss = criterion(predictions, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / num_batches
            print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return model


def evaluate_model(model, test_loader):
    """Evaluate the Time-Embedded RNN model."""
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_data, labels in test_loader:
            predictions = model(batch_data)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    binary_predictions = (all_predictions > 0.5).astype(int)

    return all_labels, all_predictions, binary_predictions


def main():
    """Main training and evaluation loop."""
    print("="*80)
    print("TIME-EMBEDDED RNN MODEL FOR AKI PREDICTION")
    print("="*80)

    # Load data
    patients = load_and_prepare_patients()

    # Metrics storage
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    auc_scores = []
    specificity_scores = []
    auc_pr_scores = []

    # ROC curve plot
    plt.figure(figsize=(10, 8))

    # Get all temporal features
    print("\nExtracting temporal features...")
    all_temporal_features = get_all_temporal_features(patients)
    input_dim = len(all_temporal_features)
    print(f"Number of temporal features: {input_dim}")

    # Cross-validation
    for fold, (train_patients, test_patients) in enumerate(trainTestPatients(patients)):
        print(f"\n{'='*80}")
        print(f"FOLD {fold}")
        print(f"{'='*80}")

        # Create datasets
        train_dataset = IrregularTimeSeriesDataset(train_patients, all_temporal_features)

        if len(train_dataset) == 0:
            print(f"Skipping fold {fold}: Empty train dataset")
            continue

        norm_stats = train_dataset.get_normalization_stats()
        test_dataset = IrregularTimeSeriesDataset(test_patients, all_temporal_features, norm_stats)

        if len(test_dataset) == 0:
            print(f"Skipping fold {fold}: Empty test dataset")
            continue

        print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        # Create model
        model = TimeEmbeddedRNNModel(
            input_dim=input_dim,
            hidden_dim=128,
            time_dim=32,
            output_dim=1
        ).to(DEVICE)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train
        print("\nTraining...")
        model = train_model(model, train_loader, criterion, optimizer, num_epochs=60)

        # Evaluate
        print("\nEvaluating...")
        y_test, y_pred_proba, y_pred = evaluate_model(model, test_loader)

        # Calculate metrics
        tn, fp, _, _ = confusion_matrix(y_test, y_pred).ravel()
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        specificity_scores.append(tn / (tn + fp))
        precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_pred_proba))
        auc_pr_scores.append(auc(recall_vals, precision_vals))

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC = {auc_scores[-1]:.3f})")

        print(f"Fold {fold} - AUC: {auc_scores[-1]:.3f}, Accuracy: {accuracy_scores[-1]:.3f}")

    # Plot ROC curves
    plt.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2, label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Time-Embedded RNN")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig("result/roc_time_embedded_rnn.png", dpi=300, bbox_inches="tight")
    print("\nSaved ROC plot to: result/roc_time_embedded_rnn.png")
    plt.show()

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY (Time-Embedded RNN)")
    print("="*80)
    print(f"AUC:         {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"Accuracy:    {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    print(f"Specificity: {np.mean(specificity_scores):.4f} ± {np.std(specificity_scores):.4f}")
    print(f"Precision:   {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Recall:      {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(f"AUC-PR:      {np.mean(auc_pr_scores):.4f} ± {np.std(auc_pr_scores):.4f}")
    print("="*80)

    print("\nComparison:")
    print("  MLP (last only):       AUC 0.770")
    print("  ODE-RNN:               AUC 0.746")
    print("  MLP (time-aware):      AUC 0.771")
    print(f"  Time-Embedded RNN:     AUC {np.mean(auc_scores):.3f}")


if __name__ == "__main__":
    main()
