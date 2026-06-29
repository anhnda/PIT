# Data Leakage Fix — Landmark Prediction Protocol

This document describes the changes made to remove an **outcome-dependent
windowing leak** in the AKI prediction pipeline (PIT), the same issue raised by
**Reviewer 6, point 2** of the BSPC submission, plus a second, deeper leak
through the renal-marker features that was not explicitly named by the
reviewers but shares the same root cause.

## 1. What was wrong

### 1a. Outcome-dependent observation windows (Reviewer 6.2)
The cohort extraction used `getUntilAkiPositive=True`. For AKI-positive
patients this truncated the observation window at each patient's individual
onset time (`akdTime`):

```python
# class_patient.py (old)
x.getMeasuresBetween(fromTime, x.akdTime if x.akdTime < toTime else toTime, ...)
```

while negative patients were observed over the full window. As a result:

- **Sequence length and measurement density depended on the label.** A short
  window / sparse tail correlated with being positive. The model could infer
  AKI status from observation duration alone.

### 1b. Renal markers sampled at the diagnosis threshold (deeper leak)
AKI/AKD is **defined** from serum creatinine and urine output
(`akdTime = min(akdCreatTime, akdUrineTime)`). Because the window ended exactly
at `akdTime` for positives, the **last-observed** serum creatinine / eGFR fed
to the classifier (`Last_i`, which includes `scr` and `egfr`) was sampled right
at the moment creatinine crossed the diagnostic threshold. The input therefore
contained the label-defining quantity almost directly. This is why plain
TabPFN reached ~0.85–0.90 AUPR: it was reading the diagnosis, not predicting it.

A second contributing bug (now also fixed) inflated nothing but *masked* this
leak's effect on the baseline: a one-hot/index misalignment in
`DataNormalizer` (`prepare_data.py`) silently turned categorical columns
(e.g. `race_*`) into NaN→0, which depressed the baseline. Fixing it revealed
that the leak-driven baseline is actually as strong as or stronger than PIT.

### 1c. A mislabeled "leak prevention" block
`TimeEmbedding.extract_temporal_data` contained a comment
`# Calculate cutoff time to prevent data leakage` that did the opposite — it
truncated the window at `aki_cutoff_hours` for positives, *creating* the leak.

## 2. The fix — fixed landmark + prediction horizon

We convert the task into a proper **landmark prediction** problem, following
Reviewer 6's prescription (fixed landmark, future horizon, exclude prevalent
cases, report cohort accounting).

New method `Patients.applyLandmarkHorizon(landmark=24h, horizon=48h)`:

1. **Fixed observation window.** Every patient is observed over the same window
   `[0, landmark]`, independent of outcome. No onset-based truncation anywhere.
2. **Exclude prevalent AKI.** Patients with `akdTime <= landmark` are dropped:
   their outcome already occurred, so it is not a prediction target.
3. **Relabel on the horizon.** A patient is positive iff
   `landmark < akdTime <= landmark + horizon`; otherwise negative (no AKI, or
   AKI after the horizon).

After this, the renal markers in `Last_i` are sampled strictly **before** any
retained patient's onset, so they no longer encode the label.

### Default landmark/horizon
`landmark = 24h`, `horizon = 48h` (predict AKI in the 24–72h window).
These are parameters of `applyLandmarkHorizon` / `load_and_prepare_patients`
and can be changed for sensitivity analysis (e.g. report lead time, or sweep
12h/24h landmarks as the reviewer suggested).

## 3. Files changed

| File | Change |
|------|--------|
| `utils/class_patient.py` | Added `Patients.applyLandmarkHorizon(...)`; returns cohort accounting (excluded count, pos rate, etc.). |
| `TimeEmbedding.py` | `load_and_prepare_patients()` now applies the landmark protocol centrally. `extract_temporal_data()` no longer truncates the window at onset — fixed window for all patients. `get_all_temporal_features()` uses `getUntilAkiPositive=False`. |
| `utils/prepare_data.py` | Fixed the one-hot/index misalignment in `DataNormalizer.fit_transform` and `.transform` (reset both indices before `join`) so categorical features are no longer corrupted. |
| `TabPFNRL.py` | Baseline extraction switched to `getUntilAkiPositive=False`; reward no longer mixes a val-set scalar into the per-sample training signal (separate fix). |
| `CatBoostBase.py`, `XGBase.py`, `TabPFNBase.py` | Apply `applyLandmarkHorizon` after loading; `getUntilAkiPositive=False`. |
| `debug_fold1.py` | Applies the landmark protocol before splitting; `getUntilAkiPositive=False`. |
| `grud.py`, `grud_plus.py`, `ode.py`, `ode_plus.py`, `TimeEmbeddingVal.py` | No edit needed — they consume `load_and_prepare_patients` and `extract_temporal_data`, so they inherit the fix automatically. |

## 4. What to expect after re-running

- **Cohort shrinks.** Prevalent AKI cases (onset ≤ landmark) are removed, so
  N and the positive rate change. The exact numbers are printed by the
  `[Landmark] ...` line at load time and must be reported in the revised paper
  (Table 1 / cohort description).
- **Absolute AUC/AUPR will likely drop** for *all* methods, because the easy
  leak signal is gone. This is expected and correct — the previous numbers
  (e.g. TabPFN AUPR ~0.85) were leak-inflated.
- **The PIT-vs-baseline gap may shrink or invert.** With the categorical bug
  fixed and the leak removed, the strongest TabPFN baseline is competitive
  with or ahead of PIT on the folds tested so far. Re-run the full 5-fold
  protocol and read the mean ± std table before drawing conclusions. If PIT no
  longer beats a correctly-implemented baseline, the contribution framing needs
  to change (e.g. cross-fold stability, calibration, or a subgroup where
  temporal signal genuinely helps) — see Reviewer 6.3/6.4 and Reviewer 1.6.

## 5. Sanity check

`applyLandmarkHorizon` was unit-tested on synthetic patients:

```
onset 10h  (pos) -> excluded (pre-landmark)
onset 24h  (pos) -> excluded (== landmark)
onset 48h  (pos) -> POSITIVE (in horizon)
onset 72h  (pos) -> POSITIVE (== horizon end)
onset 100h (pos) -> negative (after horizon)
never AKI        -> negative
```

All assertions pass.

## 6. Not yet addressed (separate reviewer points)

This fix targets the leak only. Still open for the revision:
- Ablations: stochastic vs deterministic encoder, REINFORCE vs supervised,
  with/without time embeddings, reward components (Reviewer 6.3, Reviewer 1.2).
- Statistical tests / CIs / repeated seeds / calibration (Reviewer 6.4,
  Reviewer 1.8).
- More metrics beyond AUC-ROC (Reviewer 1.3).
- Fair SOTA comparison on the same cohort/task (Reviewer 1.7).
- Report prediction **lead time** (Reviewer 6.2).
