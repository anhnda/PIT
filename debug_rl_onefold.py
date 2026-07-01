"""
debug_one_fold.py  --  isolate and deep-debug ONE hold-out CV fold.

Reproduces EXACTLY the same data split, net init, and normalization that
run_holdout() in debug_rl_rloo.py produces for a given replicate index, so the
numbers here match the corresponding `foldN` block of a full --holdout run.

What it replicates (must stay bit-identical to run_holdout):
  hold-out : stratified_holdout(pl, Y_all, HOLDOUT_FRAC, HOLDOUT_SEED) -> rem
  fold     : StratifiedKFold(n_splits=REPS, shuffle=True, random_state=SEED)
             .split(rem, remY), take the FOLD_ID-th (tr_idx, va_idx)
  net seed : torch.manual_seed(FOLD_ID); np.random.seed(FOLD_ID)   # == rep
  stats    : HybridDataset(tr_list).get_normalization_stats()

Set the knobs below and run. It does NOT touch other folds and does NOT run any
significance test -- pure single-fold introspection. Hold-out is scored only for
logging (same as the main script); no selection uses it.

  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python debug_one_fold.py
"""
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold

import debug_rl_rloo as D
from debug_rl_rloo import (
    cohort_labels, stratified_holdout, make_net, rl_rloo, report,
    blocks_for_eval, fit_score,
)
from TabPFNRL import SimpleStaticEncoder, FIXED_FEATURES, HybridDataset
from TimeEmbeddingVal import get_all_temporal_features, load_and_prepare_patients

# ============================ KNOBS ============================
FOLD_ID       = 1          # which replicate to debug (0..REPS-1)

# --- these MUST match the full run you are reproducing ---
HOLDOUT_FRAC  = 0.2
HOLDOUT_SEED  = 12345
REPS          = 10
SEED          = 27         # StratifiedKFold random_state (== args.seed)
CLF           = "xgb"
ENCODER       = "final"
K             = 4
RL_EPOCHS     = 40
EVAL_EVERY    = 4
LR            = 1e-4
ENT_COEF      = 0.01
WHITEN        = True
NO_NORMALIZE  = True       # True == pass --no_normalize
STOP_METRIC   = "none"     # report() selection; hold-out untouched either way

SAVE_LOG_NPZ  = f"fold{FOLD_ID}_debug.npz"   # per-epoch dump for offline analysis
# ==============================================================


def main():
    D.NORMALIZE = not NO_NORMALIZE
    print(f"[normalize] input z-scoring = {D.NORMALIZE}", flush=True)
    print(f"[debug-one-fold] FOLD_ID={FOLD_ID}  clf={CLF}  epochs={RL_EPOCHS}  lr={LR}",
          flush=True)

    patients = load_and_prepare_patients()
    feats = get_all_temporal_features(patients)
    enc = SimpleStaticEncoder(FIXED_FEATURES); enc.fit(patients.patientList)

    # ---- replicate hold-out exactly ----
    pl = patients.patientList
    Y_all = cohort_labels(pl, feats, enc)
    ho, rem, hoY, remY = stratified_holdout(pl, Y_all, HOLDOUT_FRAC, HOLDOUT_SEED)
    print(f"[holdout] frac~{HOLDOUT_FRAC:g} seed={HOLDOUT_SEED} "
          f"N_holdout={len(ho)} (pos {int(hoY.sum())}, rate {hoY.mean():.3f}) | "
          f"N_remainder={len(rem)}  FROZEN", flush=True)

    # ---- replicate the FOLD_ID-th split exactly ----
    skf = StratifiedKFold(n_splits=REPS, shuffle=True, random_state=SEED)
    splits = list(skf.split(rem, remY))
    if not (0 <= FOLD_ID < len(splits)):
        raise SystemExit(f"FOLD_ID {FOLD_ID} out of range 0..{len(splits)-1}")
    tr_idx, va_idx = splits[FOLD_ID]
    tr_list = [rem[i] for i in tr_idx]
    val_list = [rem[i] for i in va_idx]

    trY = remY[tr_idx]; vaY = remY[va_idx]
    print(f"[fold{FOLD_ID}] train N={len(tr_list)} (pos {int(trY.sum())}, "
          f"rate {trY.mean():.3f}) | val N={len(val_list)} (pos {int(vaY.sum())}, "
          f"rate {vaY.mean():.3f})", flush=True)
    # fingerprint the split so you can confirm it matches the full run
    print(f"[fold{FOLD_ID}] tr_idx[:8]={tr_idx[:8].tolist()}  "
          f"va_idx[:8]={va_idx[:8].tolist()}  "
          f"sum(tr_idx)={int(tr_idx.sum())} sum(va_idx)={int(va_idx.sum())}",
          flush=True)

    stats = HybridDataset(tr_list, feats, enc).get_normalization_stats()

    # ---- replicate net init exactly (seed == rep == FOLD_ID) ----
    torch.manual_seed(FOLD_ID); np.random.seed(FOLD_ID)
    net = make_net(ENCODER, len(feats)).to(D.DEVICE)

    # ---- run RL with full per-epoch logging ----
    log = rl_rloo(net, tr_list, ho, feats, enc, stats,
                  clf=CLF, epochs=RL_EPOCHS, eval_every=EVAL_EVERY,
                  K=K, lr=LR, ent_coef=ENT_COEF, whiten=WHITEN, val_p=val_list)

    report(f"fold{FOLD_ID}", log, K, stop_metric=STOP_METRIC)

    # ---- dump every logged field per epoch for offline analysis ----
    if SAVE_LOG_NPZ:
        keys = sorted({k for r in log for k in r.keys()})
        arrs = {k: np.array([r.get(k, np.nan) for r in log], dtype=float)
                for k in keys}
        np.savez(SAVE_LOG_NPZ, **arrs)
        print(f"\n[saved] per-epoch log -> {SAVE_LOG_NPZ}  "
              f"(fields: {', '.join(keys)})", flush=True)

    # ---- extra introspection: val-side dZ trajectory (the ONLY valid signal) ----
    print(f"\n---- VAL-side dZ trajectory (fold{FOLD_ID}) ----")
    print(f"  {'ep':>3} | {'val_dAUC':>8} {'val_dAUPR':>9} | {'drift':>5} {'A_pos':>6}")
    for r in log:
        print(f"  {r['epoch']:>3} | {r['val_dAUC']:>+8.4f} {r['val_dAUPR']:>+9.4f} | "
              f"{r['drift']:>5.2f} {r['A_pos']:>+6.2f}")


if __name__ == "__main__":
    main()