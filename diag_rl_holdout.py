"""
debug_rl_rloo.py  --  REINFORCE upgraded with an RLOO (Reinforce-Leave-One-Out)
baseline, the right variance-reduction for a SINGLE-STEP, BLACK-BOX reward.

Why RLOO and not PPO/actor-critic
---------------------------------
This is not sequential RL. One action (sample z), one scalar reward (OOF prob
from TabPFN), no state transition, no return to bootstrap. TabPFN is
non-differentiable -- gradient never flows through it; REINFORCE only needs
grad through log pi(z), with the reward as a scalar multiplier. A PPO critic
would estimate a future return that doesn't exist here, so it degenerates.
RLOO is built exactly for black-box, single-step policy gradients.

RLOO baseline (leave-one-out is over the K SAMPLES, not over patients)
---------------------------------------------------------------------
Draw K i.i.d. samples z_1..z_K from the policy for the SAME batch. Each gets a
per-patient reward R_k (shape [N]). The baseline for sample k is the mean reward
of the OTHER K-1 samples -- vectorised, no loop, no per-patient refit:

    b_k = (sum_j R_j - R_k) / (K - 1)          # [K, N]
    A_k = R_k - b_k
    loss = -(1/K) sum_k  logp_k * A_k

This is an unbiased, lower-variance gradient than single-sample REINFORCE with a
batch-mean baseline. Cost = K reward evaluations per epoch (K full OOF fits).
Keep K small (2-4); RLOO already helps at K=2.

Reward whitening: a running mean/std of the reward across epochs stabilises the
advantage scale (vs normalising inside one batch only).

Branch: scratch only (pretrain was shown to hold bad folds underwater).
Reward: same OOF p(true class) as before, reward clf == eval clf.

Run:  python debug_rl_rloo.py --folds 8 9 --clf tabpfn --K 4 \
          --rl_epochs 80 --eval_every 4 --lr 1e-3
Tip:  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python debug_rl_rloo.py ...
"""
import argparse, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
    tr_b, tr_Y = blocks_for_eval(net, tr_p, feats, enc, stats)
    te_b, te_Y = blocks_for_eval(net, te_p, feats, enc, stats)
    a0, c0 = fit_score(tr_b, tr_Y, te_b, te_Y, ["S", "L"], clf)
    a1, c1 = fit_score(tr_b, tr_Y, te_b, te_Y, ["S", "L", "Z"], clf)
    return (a1 - a0, c1 - c0)


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
            K=4, lr=1e-3, ent_coef=0.01, whiten=True):
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
        te_dap, te_dau = test_dZ(net, tr_p, te_p, feats, enc, stats, clf)
        cur_b, _ = blocks_for_eval(net, tr_p, feats, enc, stats)
        drift = float(np.linalg.norm(cur_b["Z"] - z_ref) / np.sqrt(len(z_ref)))
        log.append(dict(epoch=ep, test_dAUPR=te_dap, test_dAUC=te_dau, drift=drift,
                        **rstats))

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
            for k in range(K):
                z, logp, mean = net(t, deterministic=False, temperature=1.0)
                logps.append(logp)           # [N], differentiable wrt policy
                Xz = torch.cat([s, z], dim=1).detach().cpu().numpy()
                R[k] = oof_reward(Xz, Y, clf, seed=k)  # different fold seed per sample

            # ---- reward whitening (running) ----
            if whiten:
                batch_mean = R.mean(); batch_var = R.var()
                run_n += 1
                run_mean += (batch_mean - run_mean) / run_n
                run_var += (batch_var - run_var) / run_n
                Rw = (R - run_mean) / (np.sqrt(run_var) + 1e-8)
            else:
                Rw = R

            # ---- RLOO baseline over the K axis (vectorised) ----
            if K > 1:
                baseline = (Rw.sum(0, keepdims=True) - Rw) / (K - 1)   # [K,N]
            else:
                baseline = np.zeros_like(Rw)
            A = Rw - baseline                                          # [K,N]

            # ---- policy gradient ----
            A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE) # [K,N]
            loss_terms = []
            for k in range(K):
                loss_terms.append(-(logps[k] * A_t[k]).mean())
            policy_loss = torch.stack(loss_terms).mean()

            # entropy on the policy's own std (use last sample's z spread as proxy)
            ent = 0.5 * torch.log(2 * np.pi * np.e * (z.var(dim=0) + 1e-6)).sum()
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
def report(fold, log, K):
    te_ap = np.array([r["test_dAUPR"] for r in log])
    te_au = np.array([r["test_dAUC"]  for r in log])
    eps   = np.array([r["epoch"]      for r in log])
    start_ap = te_ap[0]; best_i = int(np.argmax(te_ap)); final_i = len(log) - 1

    print(f"\n----  FOLD {fold}  RLOO K={K}  ----")
    print(f"  {'ep':>3} | {'test_dAUPR':>10} {'test_dAUC':>9} | {'drift':>6} | "
          f"{'r_pos':>6} {'r_neg':>6} {'A_pos':>7} {'A_neg':>7} | {'grad':>5}")
    for r in log:
        print(f"  {r['epoch']:>3} | {r['test_dAUPR']:>+10.4f} {r['test_dAUC']:>+9.4f} | "
              f"{r['drift']:>6.3f} | {r['r_pos']:>6.3f} {r['r_neg']:>6.3f} "
              f"{r['A_pos']:>+7.3f} {r['A_neg']:>+7.3f} | {r['grad']:>5.2f}")
    print(f"  start {start_ap:+.4f} | best {te_ap[best_i]:+.4f} @ep{eps[best_i]} "
          f"| final {te_ap[final_i]:+.4f} (AUPR)  ||  final AUC {te_au[final_i]:+.4f}")
    return dict(fold=fold, start=start_ap, best=te_ap[best_i],
                final_ap=te_ap[final_i], final_au=te_au[final_i])


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
        tr_obj, _ = split_patients_train_val(train_full, val_ratio=0.1, seed=42)
        tp = tr_obj.patientList
        stats = HybridDataset(tp, feats, enc).get_normalization_stats()
        torch.manual_seed(0); np.random.seed(0)
        net = make_net(args.encoder, len(feats)).to(DEVICE)
        log = rl_rloo(net, tp, test_p.patientList, feats, enc, stats,
                      clf=args.clf, epochs=args.rl_epochs, eval_every=args.eval_every,
                      K=args.K, lr=args.lr, ent_coef=args.ent_coef,
                      whiten=not args.no_whiten)
        summ.append(report(fi, log, args.K))
    print_summary(summ, "RLOO scratch, per-fold test")


def run_holdout(patients, feats, enc, args):
    pl = patients.patientList
    Y_all = cohort_labels(pl, feats, enc)
    ho, rem, hoY, remY = stratified_holdout(pl, Y_all, args.holdout_frac, args.holdout_seed)
    print(f"[holdout] frac~{args.holdout_frac:g} seed={args.holdout_seed} "
          f"N_holdout={len(ho)} (pos {int(hoY.sum())}, rate {hoY.mean():.3f}) | "
          f"N_remainder={len(rem)}  FROZEN, never seen in training", flush=True)

    summ = []
    for rep in range(args.reps):
        # SAME hold-out every replicate; only the remainder train/val carve and
        # the init seed change. Spread across replicates = training noise.
        rng = np.random.RandomState(1000 + rep)
        idx = np.arange(len(rem)); rng.shuffle(idx)
        n_val = int(len(rem) * 0.1)
        tr_list = [rem[i] for i in idx[n_val:]]
        stats = HybridDataset(tr_list, feats, enc).get_normalization_stats()

        torch.manual_seed(rep); np.random.seed(rep)
        net = make_net(args.encoder, len(feats)).to(DEVICE)
        log = rl_rloo(net, tr_list, ho, feats, enc, stats,
                      clf=args.clf, epochs=args.rl_epochs, eval_every=args.eval_every,
                      K=args.K, lr=args.lr, ent_coef=args.ent_coef,
                      whiten=not args.no_whiten)
        summ.append(report(f"rep{rep}", log, args.K))

    print_summary(summ, "RLOO scratch, FROZEN hold-out (spread = training noise)")
    fa = np.array([s['final_ap'] for s in summ]); fu = np.array([s['final_au'] for s in summ])
    print(f"\n  hold-out dZ over {args.reps} replicates:")
    print(f"    AUPR  mean {fa.mean():+.4f} ± {fa.std():.4f} | wins>0 {int((fa>0).sum())}/{len(fa)}")
    print(f"    AUC   mean {fu.mean():+.4f} ± {fu.std():.4f} | wins>0 {int((fu>0).sum())}/{len(fu)}")


def print_summary(summ, title):
    print(f"\n\n================  SUMMARY ({title})  ================")
    print(f"  {'unit':>6} | {'start_AUPR':>10} {'best_AUPR':>10} {'final_AUPR':>10} "
          f"{'final_AUC':>10} | {'AUPR>0?':>8}")
    for s in summ:
        flag = "YES" if s['final_ap'] > 0 else "no"
        print(f"  {str(s['fold']):>6} | {s['start']:>+10.4f} {s['best']:>+10.4f} "
              f"{s['final_ap']:>+10.4f} {s['final_au']:>+10.4f} | {flag:>8}")
    print("\n  Watch A_pos in the per-unit tables. Single-sample REINFORCE had")
    print("  positive advantage stuck negative (~-1.7) -> positives pushed away.")
    print("  If RLOO lifts A_pos toward 0+ and AUPR rises, variance was part of")
    print("  the problem. If A_pos stays negative, the reward is the wall.")


if __name__ == "__main__":
    main()