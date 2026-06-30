"""
debug_rl_reward.py  --  scratch-only RL, with SWITCHABLE reward shaping, to test
whether the dead-positive reward (r_pos frozen ~0.34 while r_neg ~0.83) is what
caps dZ-AUPR, and whether fixing it lifts fold 8 above the S+L baseline.

Only the RL-from-scratch branch is run (pretrain was shown to be the thing that
holds fold 8 underwater; scratch already recovered +0.033 there). Now we attack
the second bottleneck: the reward ignores the positive class.

Why base reward starves positives
----------------------------------
base reward r_i = p(true class):  positives get p(y=1) (~0.34, classifier barely
flags them), negatives get 1-p(y=1) (~0.83, easy). The policy loss is
-(logp * A).mean() with A = (r - r.mean())/r.std() over the WHOLE batch. Since
~78% of the batch is negatives sitting at 0.83, the batch mean is high, so almost
every positive lands at a NEGATIVE advantage -> REINFORCE pushes z AWAY from
whatever helped positives. AUC (global ranking, driven by the many negatives)
creeps up; AUPR (the rare positive tail) cannot.

Reward modes
------------
  base     : r = p(true class); advantage normalized over whole batch  [control]
  posw     : same r, but advantage of positives scaled by --pos_weight so their
             gradient is not drowned by the negative majority
  balanced : advantage computed PER CLASS (r - mean_c)/std_c, removing the
             prevalence imbalance from the REINFORCE baseline entirely
  ap       : rank-based reward = each sample's contribution to average precision
             (positives ranked above negatives rewarded), directly shaping AUPR

Everything else is identical to diag_rl / debug_rl_fold89: OOF inner-CV reward
classifier == eval clf, full-batch REINFORCE, entropy bonus, test dZ on held-out.

Run:  python debug_rl_reward.py --folds 8 9 --clf tabpfn \
          --reward_modes base posw balanced ap --rl_epochs 80 --eval_every 4
"""
import argparse, numpy as np, torch, torch.nn as nn
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


# ---------------------------------------------------------------- raw OOF prob
def oof_proba(Xz_static, Y, clf_name, K=5, seed=0):
    """Return per-sample OOF p(y=1). Reward modes build on this."""
    ratio = float((Y == 0).sum()) / max(int((Y == 1).sum()), 1)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    proba = np.full(len(Y), np.nan)
    for tr, va in skf.split(Xz_static, Y):
        clf = make_clf(clf_name, ratio)
        clf.fit(Xz_static[tr], Y[tr])
        proba[va] = clf.predict_proba(Xz_static[va])[:, 1]
    return proba


def ap_contrib_reward(proba, Y):
    """Rank-based reward shaping AUPR. Positive i ranked above a negative is good;
    reward of a positive = fraction of negatives it outranks (its precision-style
    contribution); reward of a negative = fraction of positives it does NOT
    outrank (penalise negatives that score high). Bounded [0,1], dense, and
    directly correlated with average precision."""
    pos = np.where(Y == 1)[0]; neg = np.where(Y == 0)[0]
    r = np.zeros(len(Y), dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return np.full(len(Y), 0.5)
    pp = proba[pos][:, None]; pn = proba[neg][None, :]
    # positive reward: fraction of negatives it outranks (higher = better placed)
    r[pos] = (pp > pn).mean(axis=1)
    # negative reward: fraction of positives that outrank it (higher = better,
    # i.e. this negative is correctly sitting below the positives)
    r[neg] = (proba[neg][:, None] < proba[pos][None, :]).mean(axis=1)
    return r


def build_advantage(proba, Y, mode, pos_weight):
    """Return advantage A (same length as Y), already centered/scaled, ready to
    multiply with logp. Also return (r_mean, r_pos, r_neg) for logging."""
    Y = Y.astype(int)
    if mode == "ap":
        r = ap_contrib_reward(proba, Y)
    else:
        r = np.where(Y == 1, proba, 1.0 - proba)

    r_mean = float(r.mean())
    r_pos = float(r[Y == 1].mean()) if (Y == 1).any() else float("nan")
    r_neg = float(r[Y == 0].mean()) if (Y == 0).any() else float("nan")

    if mode == "balanced":
        A = np.empty_like(r)
        for c in (0, 1):
            m = Y == c
            rc = r[m]
            A[m] = (rc - rc.mean()) / (rc.std() + 1e-8)
    elif mode == "posw":
        # upweight POSITIVES in the baseline so their reward is not dragged below
        # the negative-dominated mean. Weighted mean/std centers around a baseline
        # that counts positives more, lifting their advantage toward positive.
        w = np.where(Y == 1, pos_weight, 1.0)
        wmean = np.average(r, weights=w)
        wvar = np.average((r - wmean) ** 2, weights=w)
        A = (r - wmean) / (np.sqrt(wvar) + 1e-8)
    else:
        A = (r - r.mean()) / (r.std() + 1e-8)

    return A, (r_mean, r_pos, r_neg)


# ---------------------------------------------------------------- RL (scratch) with logging
def rl_scratch_logged(net, tr_p, te_p, feats, enc, stats, clf, epochs, eval_every,
                      mode, pos_weight, lr=3e-4, ent_coef=0.01):
    ref_b, _ = blocks_for_eval(net, tr_p, feats, enc, stats)
    z_ref = ref_b["Z"].copy()

    ds = HybridDataset(tr_p, feats, enc, stats)
    ld = DataLoader(ds, batch_size=len(tr_p), shuffle=False,
                    collate_fn=hybrid_collate_fn)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    log = []

    def snapshot(ep, rstats):
        net.eval()
        te_dap, te_dau = test_dZ(net, tr_p, te_p, feats, enc, stats, clf)
        cur_b, _ = blocks_for_eval(net, tr_p, feats, enc, stats)
        drift = float(np.linalg.norm(cur_b["Z"] - z_ref) / np.sqrt(len(z_ref)))
        log.append(dict(epoch=ep, test_dAUPR=te_dap, test_dAUC=te_dau, drift=drift,
                        **rstats))

    snapshot(0, dict(r_mean=np.nan, r_pos=np.nan, r_neg=np.nan, grad=np.nan))

    for ep in range(1, epochs + 1):
        net.train()
        rstats = {}
        for t, lbl, s in ld:
            s = s.to(DEVICE)
            z, logp, mean = net(t, deterministic=False, temperature=1.0)
            Xz = torch.cat([s, z], dim=1).detach().cpu().numpy()
            Y = lbl.numpy().astype(int)
            proba = oof_proba(Xz, Y, clf)
            A, (rm, rp, rn) = build_advantage(proba, Y, mode, pos_weight)
            rstats = dict(r_mean=rm, r_pos=rp, r_neg=rn)
            A = torch.tensor(A, dtype=torch.float32, device=DEVICE)
            policy_loss = -(logp * A).mean()
            ent = 0.5 * torch.log(2 * np.pi * np.e * (z.var(dim=0) + 1e-6)).sum()
            loss = policy_loss - ent_coef * ent
            opt.zero_grad(); loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if ep % eval_every == 0 or ep == epochs:
            rstats["grad"] = float(gnorm)
            snapshot(ep, rstats)
    net.eval()
    return log


# ---------------------------------------------------------------- report
def report(fold, mode, log):
    eps   = np.array([r["epoch"]      for r in log])
    te_ap = np.array([r["test_dAUPR"] for r in log])
    te_au = np.array([r["test_dAUC"]  for r in log])
    base_ap, base_au = te_ap[0], te_au[0]
    best_i = int(np.argmax(te_ap)); final_i = len(log) - 1

    print(f"\n----  FOLD {fold}  reward={mode}  ----")
    print(f"  {'ep':>3} | {'test_dAUPR':>10} {'test_dAUC':>9} | {'drift':>6} | "
          f"{'r_mean':>6} {'r_pos':>6} {'r_neg':>6} | {'grad':>5}")
    for r in log:
        print(f"  {r['epoch']:>3} | {r['test_dAUPR']:>+10.4f} {r['test_dAUC']:>+9.4f} | "
              f"{r['drift']:>6.3f} | {r['r_mean']:>6.3f} {r['r_pos']:>6.3f} "
              f"{r['r_neg']:>6.3f} | {r['grad']:>5.2f}")
    print(f"  start dZ-AUPR {base_ap:+.4f} | best {te_ap[best_i]:+.4f} @ep{eps[best_i]} "
          f"| final {te_ap[final_i]:+.4f}  (>0 = above S+L baseline)")
    # did positives ever wake up?
    rp = np.array([r["r_pos"] for r in log if not np.isnan(r["r_pos"])])
    print(f"  r_pos range over run: [{rp.min():.3f}, {rp.max():.3f}] "
          f"(base mode froze ~0.34 -> want this to rise)")
    return dict(fold=fold, mode=mode, start=base_ap, best=te_ap[best_i],
                final=te_ap[final_i], rpos_max=float(rp.max()))


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--kfold", type=int, default=10)
    pa.add_argument("--folds", type=int, nargs="+", default=[8, 9])
    pa.add_argument("--clf", default="tabpfn", choices=["xgb", "cat", "tabpfn"])
    pa.add_argument("--reward_modes", nargs="+",
                    default=["base", "posw", "balanced", "ap"],
                    choices=["base", "posw", "balanced", "ap"])
    pa.add_argument("--pos_weight", type=float, default=6.0,
                    help="positive upweight in posw mode; needs to exceed neg/pos "
                         "ratio (~3.7) to push positive advantage above zero")
    pa.add_argument("--encoder", default="final", choices=["final", "pool"])
    pa.add_argument("--pretrain_epochs", type=int, default=20,
                    help="only used to size scratch budget = rl_epochs (kept for parity)")
    pa.add_argument("--rl_epochs", type=int, default=80)
    pa.add_argument("--eval_every", type=int, default=4)
    pa.add_argument("--seed", type=int, default=27)
    args = pa.parse_args()

    patients = load_and_prepare_patients()
    feats = get_all_temporal_features(patients)
    enc = SimpleStaticEncoder(FIXED_FEATURES); enc.fit(patients.patientList)
    all_folds = list(trainTestPatients(patients, k=args.kfold, seed=args.seed))

    summ = []
    for fi in args.folds:
        train_full, test_p = all_folds[fi]
        tr_obj, _ = split_patients_train_val(train_full, val_ratio=0.1, seed=42)
        tp = tr_obj.patientList
        stats = HybridDataset(tp, feats, enc).get_normalization_stats()

        for mode in args.reward_modes:
            torch.manual_seed(0); np.random.seed(0)
            net = make_net(args.encoder, len(feats)).to(DEVICE)
            log = rl_scratch_logged(net, tp, test_p.patientList, feats, enc, stats,
                                    clf=args.clf, epochs=args.rl_epochs,
                                    eval_every=args.eval_every, mode=mode,
                                    pos_weight=args.pos_weight)
            summ.append(report(fi, mode, log))

    print("\n\n================  SUMMARY (scratch only)  ================")
    print(f"  {'fold':>4} {'reward':>9} | {'start':>8} {'best':>8} {'final':>8} | "
          f"{'rpos_max':>8} | {'final>0?':>8}")
    for s in summ:
        flag = "YES" if s['final'] > 0 else "no"
        print(f"  {s['fold']:>4} {s['mode']:>9} | {s['start']:>+8.4f} {s['best']:>+8.4f} "
              f"{s['final']:>+8.4f} | {s['rpos_max']:>8.3f} | {flag:>8}")
    print("\n  Read:")
    print("  * base is the control (r_pos frozen, AUPR capped negative on fold 8).")
    print("  * if posw/balanced/ap lift r_pos AND push fold-8 final less negative")
    print("    (toward / above 0), the dead-positive reward was the bottleneck.")
    print("  * compare 'best' vs 'final': if best >> final, the good reward also")
    print("    needs a stop (but pick the reward that makes final itself good).")


if __name__ == "__main__":
    main()