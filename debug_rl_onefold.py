"""
debug_one_fold.py  --  isolate and deep-debug ONE hold-out CV fold.

Same CLI as debug_rl_rloo.py --holdout, plus --fold_id. Reproduces EXACTLY the
data split, net init, and normalization that run_holdout() produces for that
replicate index, so the numbers match the corresponding `foldN` block of a full
--holdout run. Touches no other fold; runs no significance test.

  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python debug_one_fold.py \
      --reps 10 --K 4 --rl_epochs 40 --eval_every 4 --clf xgb --lr 1e-4 \
      --stop_metric none --no_normalize --fold_id 1
"""
import argparse, numpy as np, torch
from sklearn.model_selection import StratifiedKFold

import debug_rl_rloo as D
from debug_rl_rloo import (
    cohort_labels, stratified_holdout, make_net, rl_rloo, report,
)
from TabPFNRL import SimpleStaticEncoder, FIXED_FEATURES, HybridDataset
from TimeEmbeddingVal import get_all_temporal_features, load_and_prepare_patients


def build_argparser():
    pa = argparse.ArgumentParser()
    # ---- mirror debug_rl_rloo.main() so the same command line works ----
    pa.add_argument("--fold_id", type=int, required=True,
                    help="which replicate (0..reps-1) to debug")
    pa.add_argument("--holdout", action="store_true",
                    help="accepted for CLI parity; this script is always holdout")
    pa.add_argument("--holdout_frac", type=float, default=0.2)
    pa.add_argument("--holdout_seed", type=int, default=12345)
    pa.add_argument("--reps", type=int, default=10)
    pa.add_argument("--clf", default="xgb", choices=["xgb", "cat", "tabpfn"])
    pa.add_argument("--K", type=int, default=4)
    pa.add_argument("--encoder", default="final", choices=["final", "pool"])
    pa.add_argument("--rl_epochs", type=int, default=40)
    pa.add_argument("--eval_every", type=int, default=4)
    pa.add_argument("--lr", type=float, default=1e-4)
    pa.add_argument("--ent_coef", type=float, default=0.01)
    pa.add_argument("--no_whiten", action="store_true")
    pa.add_argument("--stop_metric", default="none", choices=["AUC", "AUPR", "none"])
    pa.add_argument("--no_normalize", action="store_true")
    pa.add_argument("--seed", type=int, default=27,
                    help="StratifiedKFold random_state (== args.seed in main)")
    pa.add_argument("--init_seed", type=int, default=None,
                    help="seed for net init (torch/np) ONLY. Default None -> uses "
                         "fold_id (matches the full run). Fix --seed and vary this "
                         "to isolate init lottery from split lottery.")
    pa.add_argument("--neutral_init", action="store_true",
                    help="zero fc_mean so Z init = 0 for every patient "
                         "(dZ at ep0 == 0 exactly; all later dZ is from RL only). "
                         "fc_logstd left untouched so exploration noise stays ~1.")
    return pa


def main():
    args = build_argparser().parse_args()
    D.NORMALIZE = not args.no_normalize
    print(f"[normalize] input z-scoring = {D.NORMALIZE}", flush=True)
    print(f"[debug-one-fold] fold_id={args.fold_id} clf={args.clf} "
          f"epochs={args.rl_epochs} lr={args.lr}", flush=True)

    patients = load_and_prepare_patients()
    feats = get_all_temporal_features(patients)
    enc = SimpleStaticEncoder(FIXED_FEATURES); enc.fit(patients.patientList)

    # ---- replicate hold-out exactly ----
    pl = patients.patientList
    Y_all = cohort_labels(pl, feats, enc)
    ho, rem, hoY, remY = stratified_holdout(pl, Y_all, args.holdout_frac,
                                            args.holdout_seed)
    print(f"[holdout] frac~{args.holdout_frac:g} seed={args.holdout_seed} "
          f"N_holdout={len(ho)} (pos {int(hoY.sum())}, rate {hoY.mean():.3f}) | "
          f"N_remainder={len(rem)}  FROZEN", flush=True)

    # ---- replicate the fold_id-th split exactly ----
    skf = StratifiedKFold(n_splits=args.reps, shuffle=True, random_state=args.seed)
    splits = list(skf.split(rem, remY))
    if not (0 <= args.fold_id < len(splits)):
        raise SystemExit(f"--fold_id {args.fold_id} out of range 0..{len(splits)-1}")
    tr_idx, va_idx = splits[args.fold_id]
    tr_list = [rem[i] for i in tr_idx]
    val_list = [rem[i] for i in va_idx]

    trY = remY[tr_idx]; vaY = remY[va_idx]
    print(f"[fold{args.fold_id}] train N={len(tr_list)} (pos {int(trY.sum())}, "
          f"rate {trY.mean():.3f}) | val N={len(val_list)} (pos {int(vaY.sum())}, "
          f"rate {vaY.mean():.3f})", flush=True)
    print(f"[fold{args.fold_id}] tr_idx[:8]={tr_idx[:8].tolist()}  "
          f"va_idx[:8]={va_idx[:8].tolist()}  "
          f"sum(tr_idx)={int(tr_idx.sum())} sum(va_idx)={int(va_idx.sum())}",
          flush=True)

    stats = HybridDataset(tr_list, feats, enc).get_normalization_stats()

    # ---- net init: init_seed if given, else fold_id (matches full run) ----
    init_seed = args.fold_id if args.init_seed is None else args.init_seed
    torch.manual_seed(init_seed); np.random.seed(init_seed)
    print(f"[fold{args.fold_id}] net init_seed={init_seed} "
          f"(split seed={args.seed})", flush=True)
    net = make_net(args.encoder, len(feats)).to(D.DEVICE)

    if args.neutral_init:
        # zero the mean head -> mean = fc_mean(h) = 0 for every patient at ep0,
        # so Z init is a constant-0 vector and dZ(ep0) == 0 exactly. fc_logstd is
        # left as-is so the Gaussian policy still has ~unit std to explore, and
        # gradients still flow through fc_mean (via log_prob) so RL escapes 0.
        if not hasattr(net, "fc_mean"):
            raise SystemExit("--neutral_init: net has no fc_mean; "
                             "check encoder architecture")
        torch.nn.init.zeros_(net.fc_mean.weight)
        torch.nn.init.zeros_(net.fc_mean.bias)
        print(f"[fold{args.fold_id}] neutral_init: fc_mean zeroed "
              f"(Z init = 0, dZ@ep0 should be 0)", flush=True)

    # ---- run RL with full per-epoch logging ----
    log = rl_rloo(net, tr_list, ho, feats, enc, stats,
                  clf=args.clf, epochs=args.rl_epochs, eval_every=args.eval_every,
                  K=args.K, lr=args.lr, ent_coef=args.ent_coef,
                  whiten=not args.no_whiten, val_p=val_list)

    report(f"fold{args.fold_id}", log, args.K, stop_metric=args.stop_metric)


if __name__ == "__main__":
    main()