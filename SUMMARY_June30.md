# PIT / AKI-prediction investigation — summary of findings

State as of now. All numbers are out-of-fold (OOF) inside train, 5-fold,
mean±std, on the leak-free landmark cohort. `[OK]` means |mean| > std (survives
fold noise); `[noise]` means it does not.

## 0. Cohort (after fixes)
- Landmark protocol: observe [0, 24h], predict AKI in (24h, 72h], exclude
  patients with AKI onset <= 24h.
- Cohort: 1206 -> 997 patients (209 prevalent-AKI excluded).
- Positive rate: 39.2% (leaky) -> **21.8%** (leak-free).

## 1. The leak (confirmed, fixed)
Two coupled bugs, both fixed:
- **One-hot / index misalignment** in `DataNormalizer` (prepare_data.py): a
  `.join()` on a duplicated index (getMeasuresBetween concats per-patient rows
  without ignore_index) corrupted categorical columns (race etc.) into NaN->0.
  This silently depressed the baseline, making PIT look better than it was.
- **Outcome-dependent windowing** (`getUntilAkiPositive=True`): positives were
  observed only up to their AKI onset, so sequence length, measurement density,
  AND last-observed creatinine/eGFR (the label-defining variables) all leaked
  the label. This is exactly Reviewer 6 point 2.
  - Quantified: positives had 58% of measurements inside [-6,24]h vs 85% for
    negatives — a class-dependent density bias.
  - Fixed via landmark protocol + fixed window. Temporal sequences remain rich
    after the fix (median ~40-47 timestamps/patient, 0% with <3 obs), so the
    fix does NOT gut temporal info.

**Effect of fixing the leak:** AUPR drops from ~0.81 (paper) to ~0.52 (clean).
That ~0.29 drop is the size of the leak. The paper's headline numbers were
leak-inflated. After the fix, plain TabPFN on last+static is competitive with
or ahead of the original PIT — the paper's main claim does not survive on
clean data.

## 2. The encoder: original design fails, statistical pooling works
The original RNN keeps only the FINAL hidden state -> recency-biased -> it
learns something close to `last` (already an input feature), so the learned z
adds ~nothing.

| condition | dZ = (S+L+Z)-(S+L), AUPR | flag |
|---|---|---|
| original encoder (final state) | ~0 (-0.008 / +0.003) | dead |
| **statistical-pooling encoder** | **+0.05 to +0.09** | **[OK], 3 classifiers** |

Pooling = aggregate ALL valid hidden states into [mean‖std‖max‖last], then
project to latent. This gives the encoder capacity to learn whole-sequence
statistics instead of just the last step.

Confirmed across XGBoost, CatBoost, TabPFN (consistent):
- `Z over last+static` (S+L+Z)-(S+L): **+0.076 / +0.086 / +0.077 AUPR [OK]**
- `Z beyond hand-stats` (S+L+MS+Z)-(S+L+MS): +0.069 / +0.056 / +0.069 [OK]
- `Z replaces last` (S+Z)-(S+L): +0.077 / +0.084 / +0.076 [OK]
- `Z alone` vs (S+L): noise — z complements static, does NOT replace it.

## 3. Capacity control (it's the structure, not the parameters)
Pooled encoder (hidden=20, **18,396 params**) vs original encoder
(hidden=80, **45,096 params**):
- `Z over last+static`: pool +0.076/+0.086 vs final +0.059/+0.054 AUPR
- `Z alone` AUC: pool 0.77/0.79 vs final 0.62/0.62
Pool wins with **less than half the parameters** -> the gain is the pooling
structure, not capacity. Stuffing neurons into the final hidden state does not
fix the recency bias.

## 4. Hand-made stats are a strong, cheap competitor
- `hand-stats value` (S+L+MS)-(S+L): +0.036 / +0.035 / +0.019 AUPR [OK]
  (MS = per-feature mean+std over the window)
So temporal info is real and even np.mean/np.std capture much of it. The pooled
encoder's z adds a bit MORE on top of MS ([OK]), which is the justification for
learning over hand-engineering — but the margin is modest.

## 5. Surrogate-head strength: a sweet spot, not "weaker is better"
Pretraining the SAME pooled encoder through heads of different strength, then
measuring transfer dZ:

| head | dZ AUPR (xgb) | dZ AUPR (tabpfn) |
|---|---|---|
| linear (logistic) | +0.028 | +0.017 |
| **weak (paper's MLP)** | **+0.076** | **+0.077** |
| strong (512-wide deep) | +0.038 | +0.032 |

Inverted-U: a mid-strength head transfers best. Too weak (linear) under-shapes
z; too strong overfits / carries the classification itself so the encoder
isn't forced to learn. NOTE: this is 3 points, not a full curve, and the
"strong is worse" arm is confounded with overfitting at n=718 — needs a wider
sweep + train-fit column before it's a paper-grade finding. **This is where the
next experiment should go (a proper head-class sweep).**

## 6. RL does not help — and from-scratch can't even match pretrain
All RL here uses the fixed loop: out-of-fold reward (not in-context, which
saturates), proper entropy term, no temperature floor, no val-leak.

| condition | dZ AUPR | flag |
|---|---|---|
| pooled + supervised | **+0.083 ± 0.035** | [OK] |
| pooled + supervised + RL | +0.044 ± 0.025 | [OK] |
| pooled + RL-from-scratch | +0.017 ± 0.041 | noise |

- RL fine-tune vs pretrain: **-0.038 ± 0.019 [OK]** — RL consistently *hurts*.
- RL-from-scratch vs pretrain: **-0.066 ± 0.048 [OK]** — worse, and high-variance
  (per-fold: -0.025, +0.091, +0.027, -0.014, +0.005 — a lottery on seed/fold).

Mechanism: supervised pretrain already reaches a near-optimal z; RL sampling
around a good point with a noisy reward only diffuses it. From scratch, RL has
the capacity to occasionally find a great z (fold 1 beat pretrain) but cannot
do so reliably — the problem is variance, not capability. Either way, RL does
not earn a place as a primary contribution on this task.

## 7. Where this leaves the paper
The original framing ("policy-gradient coupling improves AKI prediction") does
not survive clean data. But a stronger, honest story is fully supported:

**Contribution = statistical-pooling temporal encoder for irregular ICU series**
- learned temporal representation adds +0.05-0.09 AUPR over last+static,
  [OK] across 3 classifiers, with capacity control;
- adds beyond hand-made temporal statistics;
- RL ablation is a clean negative result (answers Reviewer 6.3);
- leak fixed via landmark protocol (answers Reviewer 6.2);
- improvement is now larger and statistically cleaner than the original
  +0.98% (answers Reviewer 1.6 / 6.4).

## 8. Open / next
- **Head-class sweep (next experiment):** widen head sizes (32/64/128/256/512),
  vary dropout, add a train-fit column to separate "head carries it" from
  "head overfits". Turn the 3-point inverted-U into a real curve. This may be a
  secondary finding (representation regularization via surrogate strength).
- Probe z-quality (Z-alone) vs head strength to test the "encoder is forced to
  learn" mechanism directly.
- Multi-classifier reward for RL (average prob over xgb+cat+tabpfn) as a last
  attempt to make RL learn a *universal* z — only if RL is worth pursuing at all.
- Statistical tests / calibration / lead-time reporting for the revision
  (Reviewer 1.3/1.8, Reviewer 6.4).