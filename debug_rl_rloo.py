"""
debug_rl_rloo.py  --  REINFORCE with K-sample gradient averaging + cross-patient
advantage, the right variance-reduction for a SINGLE-STEP, BLACK-BOX reward.

Why not PPO / actor-critic
--------------------------
This is not sequential RL. One action (sample z), one scalar reward (OOF prob
from TabPFN), no state transition, no return to bootstrap. TabPFN is
non-differentiable -- gradient never flows through it; REINFORCE only needs grad
through log pi(z), with the reward as a scalar multiplier.

Why NOT RLOO-over-K (the bug in the first version)
--------------------------------------------------
The first cut used a leave-one-out baseline over the K samples:
b_k = mean of the other K-1 samples' rewards. But the K samples are drawn from
the SAME policy on the SAME batch, so they are near-identical -> R_k - mean(R_j)
collapsed to ~0, and the advantage (A_pos, A_neg) printed as 0.000 every epoch.
That cancelled exactly the signal we need (per-patient reward difference between
positives and negatives), so RL learned almost nothing (grad ~0.1).

Corrected scheme
----------------
Baseline is CROSS-PATIENT, per sample: A_k = (R_k - mean_n R_k)/std_n R_k (or a
running whiten across epochs). This keeps the between-patient signal. The K
samples then only AVERAGE the gradient, cutting single-sample MC variance:

    A_k = whiten_patients(R_k)              # [N], keeps pos/neg structure
    loss = mean_k [ -(logp_k * A_k).mean() ]

Cost = K reward evaluations / epoch. Keep K small (2-4).

Honest caveat: r_pos stays ~0.33 (TabPFN OOF barely flags positives), so even
with correct variance reduction the positive advantage may stay negative -- that
is the reward wall from every prior debug, not something K-sampling fixes. This
script isolates whether VARIANCE was also hurting.

Branch: scratch only. Reward clf == eval clf.

Run:  python debug_rl_rloo.py --folds 8 9 --clf tabpfn --K 4 --rl_epochs 80 \
          --eval_every 4 --lr 1e-3
      python debug_rl_rloo.py --holdout --reps 5 --clf tabpfn --K 4
Tip:  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python debug_rl_rloo.py ...
"""
import argparse, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats as sp_stats

from TabPFNRL import (
    FIXED_FEATURES, RNNPolicyNetwork, HybridDataset,
    hybrid_collate_fn, SimpleStaticEncoder,
)
from pooled_encoder import PooledRNNPolicyNetwork
from TimeEmbedding import DEVICE
from TimeEmbeddingVal import (
    get_all_temporal_features, split_patients_train_val, load_and_prepare_patients,
)
from utils.prepare_data import trainTestPatients


# ---------------------------------------------------------------- classifiers
def make_clf(name, ratio):
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=ratio, random_state=42, eval_metric="auc")
    if name == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05,
                                  loss_function="Logloss", eval_metric="AUC",
                                  scale_pos_weight=ratio, random_seed=42,
                                  verbose=False, allow_writing_files=False, task_type="CPU")
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
    raise ValueError(name)


def make_net(enc_kind, n_feat):
    Net = PooledRNNPolicyNetwork if enc_kind == "pool" else RNNPolicyNetwork
    return Net(input_dim=n_feat, hidden_dim=20, latent_dim=28, time_dim=32)


# ---------------------------------------------------------------- frozen hold-out
def cohort_labels(patientList, feats, enc):
    """One block pass over the whole cohort to get labels, in list order."""
    ds = HybridDataset(patientList, feats, enc)
    ld = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=hybrid_collate_fn)
    Y = []
    for _, lbl, _ in ld:
        Y.extend(lbl.numpy().tolist())
    return np.array(Y, dtype=int)


def stratified_holdout(patientList, Y, frac, seed):
    """Carve a stratified hold-out (never seen in training). Returns
    (holdout_list, remainder_list, holdout_Y, remainder_Y)."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(patientList))
    hold = []
    for c in (0, 1):
        ci = idx[Y == c]; rng.shuffle(ci)
        n_h = int(round(len(ci) * frac))
        hold.extend(ci[:n_h].tolist())
    hold = set(hold)
    ho  = [patientList[i] for i in idx if i in hold]
    rem = [patientList[i] for i in idx if i not in hold]
    hoY = Y[[i for i in idx if i in hold]]
    remY = Y[[i for i in idx if i not in hold]]
    return ho, rem, hoY, remY


# ---------------------------------------------------------------- blocks (S,L,Z)
def blocks_for_eval(net, patients, feats, enc, stats):
    ds = HybridDataset(patients, feats, enc, stats)
    ld = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
    S, L, Z, Y = [], [], [], []
    with torch.no_grad():
        for t, lbl, s in ld:
            _, _, mean = net(t, deterministic=True)
            Z.append(mean.cpu().numpy())
            vals = t['values'].cpu().numpy(); masks = t['masks'].cpu().numpy()
            for i in range(len(vals)):
                last = []
                for f in range(vals.shape[2]):
                    idx = np.where(masks[i, :, f] > 0)[0]
                    last.append(vals[i, idx[-1], f] if len(idx) else 0.0)
                L.append(last)
            S.append(s.numpy()); Y.extend(lbl.numpy())
    return {"S": np.vstack(S), "L": np.array(L), "Z": np.vstack(Z)}, np.array(Y)


def fit_score(tr_b, tr_Y, te_b, te_Y, spec, clf_name):
    Xtr = np.hstack([tr_b[k] for k in spec]); Xte = np.hstack([te_b[k] for k in spec])
    ratio = float((tr_Y == 0).sum()) / max(int((tr_Y == 1).sum()), 1)
    c = make_clf(clf_name, ratio); c.fit(Xtr, tr_Y)
    p = c.predict_proba(Xte)[:, 1]
    return average_precision_score(te_Y, p), roc_auc_score(te_Y, p)


def test_dZ(net, tr_p, te_p, feats, enc, stats, clf):
    """Returns dict with dZ AND absolute base/full for AUPR+AUC, plus the train
    Z block so the caller can compute drift without a second forward pass."""
    tr_b, tr_Y = blocks_for_eval(net, tr_p, feats, enc, stats)
    te_b, te_Y = blocks_for_eval(net, te_p, feats, enc, stats)
    a0, c0 = fit_score(tr_b, tr_Y, te_b, te_Y, ["S", "L"], clf)
    a1, c1 = fit_score(tr_b, tr_Y, te_b, te_Y, ["S", "L", "Z"], clf)
    return dict(dAUPR=a1 - a0, dAUC=c1 - c0,
                base_AUPR=a0, base_AUC=c0, full_AUPR=a1, full_AUC=c1,
                trainZ=tr_b["Z"])


# ---------------------------------------------------------------- reward
def oof_reward(Xz_static, Y, clf_name, K=5, seed=0):
    """OOF p(true class). Returns reward in [0,1] per patient."""
    ratio = float((Y == 0).sum()) / max(int((Y == 1).sum()), 1)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    proba = np.full(len(Y), np.nan)
    for tr, va in skf.split(Xz_static, Y):
        clf = make_clf(clf_name, ratio)
        clf.fit(Xz_static[tr], Y[tr])
        proba[va] = clf.predict_proba(Xz_static[va])[:, 1]
    return np.where(Y == 1, proba, 1.0 - proba)


# ---------------------------------------------------------------- RLOO RL (scratch)
def rl_rloo(net, tr_p, te_p, feats, enc, stats, clf, epochs, eval_every,
            K=4, lr=1e-3, ent_coef=0.01, whiten=True, val_p=None):
    ref_b, _ = blocks_for_eval(net, tr_p, feats, enc, stats)
    z_ref = ref_b["Z"].copy()

    ds = HybridDataset(tr_p, feats, enc, stats)
    ld = DataLoader(ds, batch_size=len(tr_p), shuffle=False,
                    collate_fn=hybrid_collate_fn)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # running stats for reward whitening
    run_mean, run_var, run_n = 0.0, 1.0, 0
    log = []

    def snapshot(ep, rstats):
        net.eval()
        m = test_dZ(net, tr_p, te_p, feats, enc, stats, clf)   # te_p = HOLD-OUT
        drift = float(np.linalg.norm(m["trainZ"] - z_ref) / np.sqrt(len(z_ref)))
        # VAL dZ -- the ONLY signal allowed to pick the epoch (hold-out untouched)
        if val_p is not None:
            mv = test_dZ(net, tr_p, val_p, feats, enc, stats, clf)
            val_dap, val_dau = mv["dAUPR"], mv["dAUC"]
        else:
            val_dap, val_dau = np.nan, np.nan
        log.append(dict(epoch=ep, test_dAUPR=m["dAUPR"], test_dAUC=m["dAUC"],
                        base_AUPR=m["base_AUPR"], base_AUC=m["base_AUC"],
                        full_AUPR=m["full_AUPR"], full_AUC=m["full_AUC"],
                        val_dAUPR=val_dap, val_dAUC=val_dau,
                        drift=drift, **rstats))

    snapshot(0, dict(r_mean=np.nan, r_pos=np.nan, r_neg=np.nan,
                     grad=np.nan, A_pos=np.nan, A_neg=np.nan))

    for ep in range(1, epochs + 1):
        net.train()
        rstats = {}
        for t, lbl, s in ld:
            s = s.to(DEVICE)
            Y = lbl.numpy().astype(int)
            N = len(Y)

            # ---- draw K samples, collect logp and rewards ----
            logps = []                       # list of [N] tensors (keep graph)
            R = np.empty((K, N), dtype=float)
            z_last = None
            for k in range(K):
                z, logp, mean = net(t, deterministic=False, temperature=1.0)
                logps.append(logp)           # [N], differentiable wrt policy
                z_last = z
                Xz = torch.cat([s, z], dim=1).detach().cpu().numpy()
                R[k] = oof_reward(Xz, Y, clf, seed=k)  # different fold seed per sample

            # ---- advantage = CROSS-PATIENT baseline, per sample ----
            # (the signal we must keep is between patients: a positive whose z
            #  earns higher reward than the patient-mean should be reinforced.
            #  Leaving-one-out over the K near-identical samples would cancel
            #  this to ~0 -- that was the bug. K is used only to average the
            #  gradient and cut variance, NOT to form the baseline.)
            if whiten:
                # running mean/std over per-patient reward, across epochs
                batch_mean = R.mean(); batch_var = R.var()
                run_n += 1
                run_mean += (batch_mean - run_mean) / run_n
                run_var += (batch_var - run_var) / run_n
                A = (R - run_mean) / (np.sqrt(run_var) + 1e-8)         # [K,N]
            else:
                # center each sample by its own cross-patient mean
                A = (R - R.mean(axis=1, keepdims=True)) / \
                    (R.std(axis=1, keepdims=True) + 1e-8)             # [K,N]

            # ---- policy gradient, averaged over K samples ----
            A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE) # [K,N]
            loss_terms = [-(logps[k] * A_t[k]).mean() for k in range(K)]
            policy_loss = torch.stack(loss_terms).mean()

            ent = 0.5 * torch.log(2 * np.pi * np.e * (z_last.var(dim=0) + 1e-6)).sum()
            loss = policy_loss - ent_coef * ent

            opt.zero_grad(); loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            # logging on raw reward (mean over K samples)
            Rm = R.mean(0)
            rstats = dict(r_mean=float(R.mean()),
                          r_pos=float(Rm[Y == 1].mean()) if (Y == 1).any() else np.nan,
                          r_neg=float(Rm[Y == 0].mean()) if (Y == 0).any() else np.nan,
                          A_pos=float(A[:, Y == 1].mean()) if (Y == 1).any() else np.nan,
                          A_neg=float(A[:, Y == 0].mean()) if (Y == 0).any() else np.nan)
        if ep % eval_every == 0 or ep == epochs:
            rstats["grad"] = float(gnorm)
            snapshot(ep, rstats)
    net.eval()
    return log


# ---------------------------------------------------------------- report
def report(fold, log, K, stop_metric="AUC"):
    """Pick the epoch by VAL dZ (early-stop signal), then report the HOLD-OUT dZ
    AT that epoch. Hold-out never participates in epoch selection."""
    val_key = "val_dAUC" if stop_metric == "AUC" else "val_dAUPR"
    val = np.array([r[val_key] for r in log])
    # epoch 0 is random-init; allow it to be the pick too (no-train baseline)
    if np.all(np.isnan(val)):
        stop_i = len(log) - 1                      # no val -> last epoch
    else:
        stop_i = int(np.nanargmax(val))
    final_i = len(log) - 1
    sel = log[stop_i]; fin = log[final_i]

    print(f"\n----  FOLD {fold}  RLOO K={K}  (stop on val-{stop_metric})  ----")
    print(f"  {'ep':>3} | {'val_dAUC':>8} {'val_dAUPR':>9} | "
          f"{'ho_dAUC':>8} {'ho_dAUPR':>9} | {'ho_full_AUC':>11} | {'A_pos':>6} {'drift':>5}")
    for i, r in enumerate(log):
        mark = " <== stop" if i == stop_i else ""
        print(f"  {r['epoch']:>3} | {r['val_dAUC']:>+8.4f} {r['val_dAUPR']:>+9.4f} | "
              f"{r['test_dAUC']:>+8.4f} {r['test_dAUPR']:>+9.4f} | "
              f"{r['full_AUC']:>11.4f} | {r['A_pos']:>+6.2f} {r['drift']:>5.2f}{mark}")
    print(f"  SELECTED ep{sel['epoch']} by val-{stop_metric}: "
          f"HOLD-OUT  S+L AUC {sel['base_AUC']:.4f} AUPR {sel['base_AUPR']:.4f} | "
          f"S+L+Z AUC {sel['full_AUC']:.4f} AUPR {sel['full_AUPR']:.4f} | "
          f"dZ AUC {sel['test_dAUC']:+.4f} dZ AUPR {sel['test_dAUPR']:+.4f}")
    print(f"  (for reference, FINAL ep{fin['epoch']}: dZ AUC {fin['test_dAUC']:+.4f} "
          f"dZ AUPR {fin['test_dAUPR']:+.4f})")
    return dict(fold=fold, stop_ep=sel['epoch'],
                base_ap=sel['base_AUPR'], base_au=sel['base_AUC'],
                full_ap=sel['full_AUPR'], full_au=sel['full_AUC'],
                final_ap=sel['test_dAUPR'], final_au=sel['test_dAUC'],
                final_ep_ap=fin['test_dAUPR'], final_ep_au=fin['test_dAUC'])


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--kfold", type=int, default=10)
    pa.add_argument("--folds", type=int, nargs="+", default=[8, 9],
                    help="(fold mode) ignored when --holdout is set")
    pa.add_argument("--holdout", action="store_true",
                    help="frozen stratified hold-out + CV-replicates on remainder")
    pa.add_argument("--holdout_frac", type=float, default=0.2)
    pa.add_argument("--holdout_seed", type=int, default=12345)
    pa.add_argument("--reps", type=int, default=5)
    pa.add_argument("--clf", default="tabpfn", choices=["xgb", "cat", "tabpfn"])
    pa.add_argument("--K", type=int, default=4, help="policy samples for RLOO")
    pa.add_argument("--encoder", default="final", choices=["final", "pool"])
    pa.add_argument("--rl_epochs", type=int, default=80)
    pa.add_argument("--eval_every", type=int, default=4)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--ent_coef", type=float, default=0.01)
    pa.add_argument("--no_whiten", action="store_true")
    pa.add_argument("--stop_metric", default="AUC", choices=["AUC", "AUPR"],
                    help="val metric used to pick the early-stop epoch")
    pa.add_argument("--seed", type=int, default=27)
    args = pa.parse_args()

    patients = load_and_prepare_patients()
    feats = get_all_temporal_features(patients)
    enc = SimpleStaticEncoder(FIXED_FEATURES); enc.fit(patients.patientList)

    if args.holdout:
        run_holdout(patients, feats, enc, args)
    else:
        run_folds(patients, feats, enc, args)


def run_folds(patients, feats, enc, args):
    all_folds = list(trainTestPatients(patients, k=args.kfold, seed=args.seed))
    summ = []
    for fi in args.folds:
        train_full, test_p = all_folds[fi]
        tr_obj, val_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42)
        tp = tr_obj.patientList
        stats = HybridDataset(tp, feats, enc).get_normalization_stats()
        torch.manual_seed(0); np.random.seed(0)
        net = make_net(args.encoder, len(feats)).to(DEVICE)
        log = rl_rloo(net, tp, test_p.patientList, feats, enc, stats,
                      clf=args.clf, epochs=args.rl_epochs, eval_every=args.eval_every,
                      K=args.K, lr=args.lr, ent_coef=args.ent_coef,
                      whiten=not args.no_whiten, val_p=val_obj.patientList)
        summ.append(report(fi, log, args.K, stop_metric=args.stop_metric))
    print_summary(summ, f"RLOO scratch, per-fold, early-stop on val-{args.stop_metric}")


def run_holdout(patients, feats, enc, args):
    pl = patients.patientList
    Y_all = cohort_labels(pl, feats, enc)
    ho, rem, hoY, remY = stratified_holdout(pl, Y_all, args.holdout_frac, args.holdout_seed)
    print(f"[holdout] frac~{args.holdout_frac:g} seed={args.holdout_seed} "
          f"N_holdout={len(ho)} (pos {int(hoY.sum())}, rate {hoY.mean():.3f}) | "
          f"N_remainder={len(rem)}  FROZEN, never seen in training", flush=True)

    summ = []
    # K-fold the REMAINDER. Each replicate uses one fold as VAL (for early-stop),
    # the other folds as TRAIN. Hold-out is scored only at the val-selected epoch.
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=args.reps, shuffle=True, random_state=args.seed)
    for rep, (tr_idx, va_idx) in enumerate(skf.split(rem, remY)):
        tr_list = [rem[i] for i in tr_idx]
        val_list = [rem[i] for i in va_idx]
        stats = HybridDataset(tr_list, feats, enc).get_normalization_stats()

        torch.manual_seed(rep); np.random.seed(rep)
        net = make_net(args.encoder, len(feats)).to(DEVICE)
        log = rl_rloo(net, tr_list, ho, feats, enc, stats,
                      clf=args.clf, epochs=args.rl_epochs, eval_every=args.eval_every,
                      K=args.K, lr=args.lr, ent_coef=args.ent_coef,
                      whiten=not args.no_whiten, val_p=val_list)
        summ.append(report(f"fold{rep}", log, args.K, stop_metric=args.stop_metric))

    print_summary(summ, f"RLOO scratch, {args.reps}-fold on remainder, "
                        f"early-stop on val-{args.stop_metric}, scored on FROZEN hold-out")


def print_summary(summ, title):
    print(f"\n\n================  SUMMARY ({title})  ================")
    print(f"  {'unit':>6} {'ep':>4} | {'S+L AUC':>8} {'+Z AUC':>8} {'dZ_AU':>7} | "
          f"{'S+L AUPR':>9} {'+Z AUPR':>8} {'dZ_AP':>7} | {'final_dAU':>9}")
    for s in summ:
        print(f"  {str(s['fold']):>6} {s['stop_ep']:>4} | "
              f"{s['base_au']:>8.4f} {s['full_au']:>8.4f} {s['final_au']:>+7.4f} | "
              f"{s['base_ap']:>9.4f} {s['full_ap']:>8.4f} {s['final_ap']:>+7.4f} | "
              f"{s['final_ep_au']:>+9.4f}")

    bap = np.array([s['base_ap'] for s in summ]); fap = np.array([s['full_ap'] for s in summ])
    bau = np.array([s['base_au'] for s in summ]); fau = np.array([s['full_au'] for s in summ])
    dau_sel = np.array([s['final_au'] for s in summ])
    dau_fin = np.array([s['final_ep_au'] for s in summ])
    print(f"\n  At val-selected epoch, across {len(summ)} folds:")
    print(f"    AUC :  S+L {bau.mean():.4f}±{bau.std():.4f} | S+L+Z {fau.mean():.4f}±{fau.std():.4f}"
          f" | dZ {(fau-bau).mean():+.4f} wins {int((fau>bau).sum())}/{len(summ)}")
    print(f"    AUPR:  S+L {bap.mean():.4f}±{bap.std():.4f} | S+L+Z {fap.mean():.4f}±{fap.std():.4f}"
          f" | dZ {(fap-bap).mean():+.4f} wins {int((fap>bap).sum())}/{len(summ)}")
    print(f"\n  early-stop vs run-to-end (dZ-AUC): "
          f"selected {dau_sel.mean():+.4f} | final-epoch {dau_fin.mean():+.4f} | "
          f"gain {(dau_sel-dau_fin).mean():+.4f}")

    # ---- paired significance: S+L+Z vs S+L, at the val-selected epoch ----
    def paired(name, full, base):
        full = np.asarray(full); base = np.asarray(base); d = full - base; n = len(d)
        t_stat, t_p = sp_stats.ttest_rel(full, base)
        try:
            w_stat, w_p = sp_stats.wilcoxon(full, base); w_str = f"W={w_stat:.1f} p={w_p:.4f}"
        except ValueError as e:
            w_str = f"n/a ({e})"
        dz = d.mean() / (d.std(ddof=1) + 1e-12)
        print(f"    {name:14s} mean dZ {d.mean():+.4f} | wins {int((d>0).sum())}/{n} | "
              f"paired-t t={t_stat:+.3f} p={t_p:.4f} | Wilcoxon {w_str} | dz={dz:+.2f}")

    print(f"\n  PAIRED TEST  (S+L+Z) vs (S+L)  at val-selected epoch, {len(summ)} folds:")
    paired("AUC-ROC", fau, bau)
    paired("AUPR", fap, bap)
    print("    (dz = Cohen's d for paired diffs; |dz|>0.8 large. n folds is small,")
    print("     so Wilcoxon p floors at ~0.002 for n=10 / ~0.06 for n=5.)")


if __name__ == "__main__":
    main()