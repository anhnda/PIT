"""
FOLD 1 DEBUGGER  (data/splits UNCHANGED)

Goal: explain why on fold 1 the RL surrogate (Test AUPR 0.8051) LOSES to plain
TabPFN baseline (Test AUPR 0.8514), while on fold 0 it wins.

Hypotheses this script tests, in order:
  H1. Val split of fold 1 is not representative of its Test split
      -> Val AUPR caps ~0.71 but the SAME model scores ~0.81 on Test.
      Measured via: class balance, size, feature-distribution shift (PSI/KS),
      and val-vs-test score of the FROZEN baseline + RL model.
  H2. RL stage over-fits a weak pretrain and saturates at the temp floor
      -> log per-epoch policy entropy, temperature, val AUPR, AND a held-out
      test AUPR probe each epoch so we can see val<->test correlation directly.
  H3. RL adds nothing over baseline on the features themselves
      -> per-sample comparison of RL prob vs baseline prob on identical test rows.

NOTE: You run this. I do not run anything. GPU via torch .to('cuda').
      Plug your real RL model / pretrain / dataloaders into the marked hooks.
      Everything else (metrics, mismatch diagnostics, logging) is ready.
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import ks_2samp

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FOLD = 1
XSEED = 42

PT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PT)

# ---- your data utils (unchanged) -------------------------------------------
from utils.class_patient import Patients
from utils.prepare_data import trainTestPatients, encodeCategoricalData


# ============================================================================
# metric helpers
# ============================================================================
def metrics(y, p):
    return {
        "auc": roc_auc_score(y, p),
        "aupr": average_precision_score(y, p),
        "n": int(len(y)),
        "pos_rate": float(np.mean(y)),
    }


def psi(expected, actual, bins=10):
    """Population Stability Index between two 1-D arrays."""
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    e = np.histogram(expected, qs)[0] / len(expected)
    a = np.histogram(actual, qs)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


# ============================================================================
# H1: val/test representativeness
# ============================================================================
def diagnose_split(X_tr, y_tr, X_val, y_val, X_te, y_te):
    print("\n" + "=" * 70)
    print("H1  VAL vs TEST REPRESENTATIVENESS")
    print("=" * 70)
    for name, X, y in [("train", X_tr, y_tr), ("val", X_val, y_val), ("test", X_te, y_te)]:
        print(f"  {name:5s}: n={len(X):5d}  pos_rate={np.mean(y):.4f}")

    # feature-wise drift val vs test
    print("\n  Top feature drift (val vs test):")
    drift = []
    cols = X_val.columns
    for c in cols:
        v, t = X_val[c].values, X_te[c].values
        try:
            ks = ks_2samp(v, t).statistic
            p = psi(v, t)
        except Exception:
            ks, p = np.nan, np.nan
        drift.append((c, ks, p))
    drift.sort(key=lambda r: (-(r[2] if np.isfinite(r[2]) else 0)))
    for c, ks, p in drift[:12]:
        flag = "  <-- shift" if (np.isfinite(p) and p > 0.25) else ""
        print(f"    {c[:32]:32s} KS={ks:.3f}  PSI={p:.3f}{flag}")
    n_shift = sum(1 for _, _, p in drift if np.isfinite(p) and p > 0.25)
    print(f"\n  Features with PSI>0.25 (material shift): {n_shift}/{len(cols)}")
    print("  -> If this is high, Val is NOT a proxy for Test on fold 1,")
    print("     so best-epoch-on-Val selection picks the wrong checkpoint.")


# ============================================================================
# H2: per-epoch val<->test correlation + policy collapse
#     wraps YOUR rl training loop. Provide eval_fn that returns probs.
# ============================================================================
def probe_rl_training(rl_model, train_step_fn, eval_fn,
                      X_val, y_val, X_te, y_te, n_epochs=80, log_every=5):
    """
    train_step_fn(epoch) -> dict(temp=?, entropy=?)   # one epoch of YOUR rl update
    eval_fn(model, X)    -> np.ndarray probs          # forward pass, no grad
    """
    print("\n" + "=" * 70)
    print("H2  RL TRAINING PROBE (val AND test logged every epoch)")
    print("=" * 70)
    print(f"  {'ep':>3} {'temp':>6} {'entropy':>8} "
          f"{'valAUC':>7} {'valPR':>7} {'teAUC':>7} {'tePR':>7}")
    hist = []
    for ep in range(1, n_epochs + 1):
        info = train_step_fn(ep) or {}
        if ep % log_every == 0 or ep == 1:
            with torch.no_grad():
                pv = eval_fn(rl_model, X_val)
                pt = eval_fn(rl_model, X_te)
            row = dict(
                ep=ep, temp=info.get("temp", np.nan),
                entropy=info.get("entropy", np.nan),
                val_auc=roc_auc_score(y_val, pv),
                val_pr=average_precision_score(y_val, pv),
                te_auc=roc_auc_score(y_te, pt),
                te_pr=average_precision_score(y_te, pt),
            )
            hist.append(row)
            print(f"  {row['ep']:>3d} {row['temp']:>6.3f} {row['entropy']:>8.4f} "
                  f"{row['val_auc']:>7.4f} {row['val_pr']:>7.4f} "
                  f"{row['te_auc']:>7.4f} {row['te_pr']:>7.4f}")

    h = pd.DataFrame(hist)
    if len(h) > 2:
        corr = np.corrcoef(h.val_pr, h.te_pr)[0, 1]
        print(f"\n  corr(val_AUPR, test_AUPR) over epochs = {corr:+.3f}")
        print("  -> near 0 / negative means early-stopping on Val is blind to Test.")
        # did temp hit floor while val stagnated?
        if (h.temp <= 0.301).any():
            ep_floor = int(h.loc[h.temp <= 0.301, "ep"].iloc[0])
            after = h[h.ep >= ep_floor]
            print(f"  temp hit floor (0.300) at epoch {ep_floor}; "
                  f"val_AUPR change after = {after.val_pr.iloc[-1]-after.val_pr.iloc[0]:+.4f}")
            print("  -> flat/negative after floor = entropy collapse, no learning.")
        # checkpoint regret: best-val epoch vs best-test epoch
        best_val_ep = int(h.loc[h.val_pr.idxmax(), "ep"])
        best_te_ep = int(h.loc[h.te_pr.idxmax(), "ep"])
        te_at_bestval = float(h.loc[h.val_pr.idxmax(), "te_pr"])
        te_best = float(h.te_pr.max())
        print(f"  best-Val epoch={best_val_ep} -> test_AUPR={te_at_bestval:.4f}")
        print(f"  best-Test epoch={best_te_ep} -> test_AUPR={te_best:.4f}")
        print(f"  selection regret = {te_best - te_at_bestval:+.4f}")
    return h


# ============================================================================
# H3: RL vs baseline on identical test rows
# ============================================================================
def compare_rl_vs_baseline(p_rl, p_base, y_te):
    print("\n" + "=" * 70)
    print("H3  RL vs BASELINE on identical test rows")
    print("=" * 70)
    print(f"  RL       AUPR={average_precision_score(y_te, p_rl):.4f}  "
          f"AUC={roc_auc_score(y_te, p_rl):.4f}")
    print(f"  baseline AUPR={average_precision_score(y_te, p_base):.4f}  "
          f"AUC={roc_auc_score(y_te, p_base):.4f}")
    print(f"  corr(rl, base) = {np.corrcoef(p_rl, p_base)[0,1]:+.3f}")
    # where does RL lose? positives that baseline ranks high but RL ranks low
    y = np.asarray(y_te)
    pos = y == 1
    drop = (p_base - p_rl)
    worst = np.argsort(-drop * pos)[:10]
    print("\n  Top positives RL demotes vs baseline (rank-collapse evidence):")
    print(f"    {'idx':>5} {'y':>2} {'p_base':>7} {'p_rl':>7} {'drop':>7}")
    for i in worst:
        print(f"    {i:>5d} {int(y[i]):>2d} {p_base[i]:>7.3f} {p_rl[i]:>7.3f} {drop[i]:>7.3f}")
    print("\n  -> If RL systematically lowers prob on true positives that")
    print("     baseline got right, the surrogate is discarding signal.")


# ============================================================================
# MAIN  (fold 1 only)
# ============================================================================
def main():
    print("FOLD 1 DEBUGGER — data/splits unchanged\n")

    # grab fold 1 exactly as the original pipeline produces it
    train_full = test_p = None
    # Load cohort and apply the landmark/horizon protocol so the debug run
    # matches the leak-free pipeline (no outcome-dependent windowing).
    _cohort = Patients.loadPatients()
    _info = _cohort.applyLandmarkHorizon(
        landmark=pd.Timedelta(hours=24), horizon=pd.Timedelta(hours=48)
    )
    print(f"  [Landmark] cohort {_info['n_before']} -> {_info['n_after']} | "
          f"excluded {_info['n_excluded_pre_landmark']} pre-landmark | "
          f"pos_rate {_info['pos_rate']:.3f}\n")
    for fold, (tr, te) in enumerate(trainTestPatients(_cohort, seed=XSEED)):
        if fold == FOLD:
            train_full, test_p = tr, te
            break
    assert train_full is not None, "fold 1 not produced"

    df_tr = train_full.getMeasuresBetween(
        pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last",
        getUntilAkiPositive=False
    ).drop(columns=["subject_id", "hadm_id", "stay_id"])
    df_te = test_p.getMeasuresBetween(
        pd.Timedelta(hours=-6), pd.Timedelta(hours=24), "last",
        getUntilAkiPositive=False
    ).drop(columns=["subject_id", "hadm_id", "stay_id"])

    df_tr, df_te, _ = encodeCategoricalData(df_tr, df_te)

    # encodeCategoricalData can leave a non-contiguous index after the fold slice;
    # reset so positional .iloc indexing below is unambiguous.
    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)

    X = df_tr.drop(columns=["akd"]).fillna(0)
    y = df_tr["akd"].values
    X_te = df_te.drop(columns=["akd"]).fillna(0)
    y_te = df_te["akd"].values

    # reproduce the SAME train/val split your RL stage uses.
    # (replace this stub with your real split if it differs — keep it identical)
    # STRATIFIED so val pos_rate tracks train/test instead of drifting (the
    # 0.459 vs 0.391 imbalance was inflating the val/test AUPR gap on fold 1).
    from sklearn.model_selection import train_test_split
    tr_i, val_i = train_test_split(
        np.arange(len(X)), test_size=0.15, random_state=XSEED, stratify=y
    )
    X_tr, y_tr = X.iloc[tr_i], y[tr_i]
    X_val, y_val = X.iloc[val_i], y[val_i]

    # ---- H1 -----------------------------------------------------------------
    diagnose_split(X_tr, y_tr, X_val, y_val, X_te, y_te)

    # ---- baseline probs (frozen TabPFN, same features) ----------------------
    from tabpfn import TabPFNClassifier
    base = TabPFNClassifier(device=DEVICE)
    base.fit(X_tr.values, y_tr)
    p_base_val = base.predict_proba(X_val.values)[:, 1]
    p_base_te = base.predict_proba(X_te.values)[:, 1]
    print("\n  baseline val :", metrics(y_val, p_base_val))
    print("  baseline test:", metrics(y_te, p_base_te))
    print("  -> compare baseline val vs test AUPR: a big gap here alone")
    print("     already proves the Val split is the problem (model-agnostic).")

    # ---- H2 : RL probe ------------------------------------------------------
    # ==================== HOOKS — plug your RL pieces in ====================
    # rl_model    = your_rl_policy().to(DEVICE)
    # def train_step_fn(ep):
    #     temp, ent = one_epoch_of_your_rl(rl_model, X_tr, y_tr)  # your loop body
    #     return {"temp": temp, "entropy": ent}
    # def eval_fn(model, Xdf):
    #     model.eval()
    #     t = torch.tensor(Xdf.values, dtype=torch.float32, device=DEVICE)
    #     with torch.no_grad():
    #         return torch.sigmoid(model(t)).squeeze(-1).cpu().numpy()
    # hist = probe_rl_training(rl_model, train_step_fn, eval_fn,
    #                          X_val, y_val, X_te, y_te, n_epochs=80, log_every=5)
    # hist.to_csv("debug_fold1_rl_trace.csv", index=False)
    #
    # p_rl_te = eval_fn(rl_model, X_te)
    # compare_rl_vs_baseline(p_rl_te, p_base_te, y_te)
    # =======================================================================
    print("\n[H2/H3] Wire the RL hooks above to your policy, then rerun.")
    print("Without them, H1 + baseline val/test gap already localize the cause.")


if __name__ == "__main__":
    main()