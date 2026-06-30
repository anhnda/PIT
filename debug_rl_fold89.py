"""
debug_rl_fold89.py  --  Why does RL fine-tuning HELP some folds and HURT others,
and can we tell (from TRAIN only) when to stop so a bad fold stops degrading?

Focus: fold 8 (RL on top of pretrain ends up well BELOW the S+L baseline,
dZ-AUPR pre -0.0629) vs fold 9 (dZ-AUPR pre +0.0247, RL stays positive).

This is a *diagnostic*, not a new training protocol. For each target fold we:

  1. supervised-pretrain the encoder (train-only, val checkpoint) -- same as diag_rl
  2. run RL fine-tune, but log EVERY epoch:
       train side (no test peeking, usable as a stopping signal):
         - reward_mean / reward_std on the (sampled) train batch
         - reward separation: mean reward on positives vs negatives
         - OOF-AUPR / OOF-AUC of (S+L+Z) on TRAIN via inner StratifiedKFold
           ("train_dZ" = that minus OOF (S+L)); this is the honest proxy
         - drift: ||z_mean - z_mean_pretrain|| (how far RL has moved z)
         - policy grad-norm
       test side (HELD OUT -- logged ONLY to see what the proxy should track):
         - real dZ-AUPR / dZ-AUC = (S+L+Z)-(S+L) on the fold's test patients

  3. report, per fold:
       - the epoch where TEST dZ peaks  (oracle, not usable online)
       - the epoch where the TRAIN proxy peaks (what an honest early-stop sees)
       - correlation(train_proxy, test_dZ) across epochs
       - what early-stopping on the train proxy WOULD have given vs running full

Read: if train-OOF-dZ tracks test-dZ, then "stop when train-OOF-dZ stops
improving" caps the damage on fold 8 without ever touching test. If reward_mean
keeps rising while train-OOF-dZ falls, that's RL over-optimising a saturating
reward -- the classic REINFORCE-eats-itself failure, and the reward (not the
fold) is the thing to fix.

Run:  python debug_rl_fold89.py --folds 8 9 --clf tabpfn --rl_epochs 60 \
                                --eval_every 2
(reward, train proxy, and final dZ all use --clf, so the reward boundary
matches the boundary dZ is scored on. tabpfn here = slow; cut rl_epochs or
raise eval_every if needed.)
"""
import argparse, copy, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats as sp_stats

from TabPFNRL import (
    FIXED_FEATURES, SupervisedHead, RNNPolicyNetwork, HybridDataset,
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


# ---------------------------------------------------------------- supervised pretrain
def supervised_pretrain(net, loader, val_loader, epochs, lr=1e-3):
    head = SupervisedHead(net.latent_dim + len(FIXED_FEATURES)).to(DEVICE)
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=lr)
    crit = nn.BCELoss()
    best_aupr, best_state = -1, None
    for _ in range(epochs):
        net.train(); head.train()
        for t, lbl, s in loader:
            lbl = lbl.to(DEVICE); s = s.to(DEVICE)
            z, _, _ = net(t, deterministic=True)
            p = head(torch.cat([z, s], dim=1)).squeeze(-1)
            loss = crit(p, lbl)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(net.parameters()) + list(head.parameters()), 1.0)
            opt.step()
        net.eval(); head.eval()
        ps, ys = [], []
        with torch.no_grad():
            for t, lbl, s in val_loader:
                s = s.to(DEVICE); z, _, _ = net(t, deterministic=True)
                p = head(torch.cat([z, s], dim=1)).squeeze(-1)
                ps.extend(p.cpu().numpy()); ys.extend(lbl.numpy())
        au = average_precision_score(ys, ps) if len(set(ys)) > 1 else 0
        if au > best_aupr:
            best_aupr = au; best_state = copy.deepcopy(net.state_dict())
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval(); return net


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


def train_oof_dZ(net, tr_p, feats, enc, stats, clf_name, K=5, seed=0):
    """Honest, test-free proxy: OOF (S+L+Z) minus OOF (S+L) on TRAIN only."""
    b, Y = blocks_for_eval(net, tr_p, feats, enc, stats)
    ratio = float((Y == 0).sum()) / max(int((Y == 1).sum()), 1)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)

    def oof(spec):
        X = np.hstack([b[k] for k in spec])
        p = np.full(len(Y), np.nan)
        for tr, va in skf.split(X, Y):
            c = make_clf(clf_name, ratio); c.fit(X[tr], Y[tr])
            p[va] = c.predict_proba(X[va])[:, 1]
        return average_precision_score(Y, p), roc_auc_score(Y, p)

    a0, c0 = oof(["S", "L"]); a1, c1 = oof(["S", "L", "Z"])
    return (a1 - a0, c1 - c0)


# ---------------------------------------------------------------- reward (OOF)
def oof_reward(Xz_static, Y, clf_name, K=5, seed=0):
    ratio = float((Y == 0).sum()) / max(int((Y == 1).sum()), 1)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    proba = np.full(len(Y), np.nan)
    for tr, va in skf.split(Xz_static, Y):
        clf = make_clf(clf_name, ratio)
        clf.fit(Xz_static[tr], Y[tr])
        proba[va] = clf.predict_proba(Xz_static[va])[:, 1]
    return np.where(Y == 1, proba, 1.0 - proba)


# ---------------------------------------------------------------- RL with logging
def rl_finetune_logged(net, tr_p, te_p, feats, enc, stats, reward_clf, eval_clf,
                       epochs, eval_every, lr=3e-4, ent_coef=0.01):
    """Run RL exactly like diag_rl, but snapshot train-proxy + test-dZ along the
    way. z_ref = pretrained mean(z) for drift. Returns a per-epoch log."""
    # reference z (pretrained, deterministic) for drift measurement
    ref_b, _ = blocks_for_eval(net, tr_p, feats, enc, stats)
    z_ref = ref_b["Z"].copy()

    ds = HybridDataset(tr_p, feats, enc, stats)
    ld = DataLoader(ds, batch_size=len(tr_p), shuffle=False,
                    collate_fn=hybrid_collate_fn)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    log = []  # list of dicts

    def snapshot(ep):
        net.eval()
        tr_dap, tr_dau = train_oof_dZ(net, tr_p, feats, enc, stats, eval_clf)
        te_dap, te_dau = test_dZ(net, tr_p, te_p, feats, enc, stats, eval_clf)
        cur_b, _ = blocks_for_eval(net, tr_p, feats, enc, stats)
        drift = float(np.linalg.norm(cur_b["Z"] - z_ref) / np.sqrt(len(z_ref)))
        log.append(dict(epoch=ep, train_dAUPR=tr_dap, train_dAUC=tr_dau,
                        test_dAUPR=te_dap, test_dAUC=te_dau, drift=drift,
                        reward_mean=np.nan, reward_pos=np.nan, reward_neg=np.nan,
                        grad=np.nan))

    snapshot(0)  # epoch 0 == pretrained state, before any RL step

    for ep in range(1, epochs + 1):
        net.train()
        rstats = {}
        for t, lbl, s in ld:
            s = s.to(DEVICE)
            z, logp, mean = net(t, deterministic=False, temperature=1.0)
            Xz = torch.cat([s, z], dim=1).detach().cpu().numpy()
            Y = lbl.numpy().astype(int)
            reward = oof_reward(Xz, Y, reward_clf)
            rstats = dict(reward_mean=float(reward.mean()),
                          reward_pos=float(reward[Y == 1].mean()),
                          reward_neg=float(reward[Y == 0].mean()))
            R = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
            R = (R - R.mean()) / (R.std() + 1e-8)
            policy_loss = -(logp * R).mean()
            ent = 0.5 * torch.log(2 * np.pi * np.e * (z.var(dim=0) + 1e-6)).sum()
            loss = policy_loss - ent_coef * ent
            opt.zero_grad(); loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if ep % eval_every == 0 or ep == epochs:
            snapshot(ep)
            log[-1].update(rstats); log[-1]["grad"] = float(gnorm)
    net.eval()
    return log


# ---------------------------------------------------------------- analysis
def analyse(fold, log):
    eps   = np.array([r["epoch"]       for r in log])
    te_ap = np.array([r["test_dAUPR"]  for r in log])
    te_au = np.array([r["test_dAUC"]   for r in log])
    tr_ap = np.array([r["train_dAUPR"] for r in log])
    tr_au = np.array([r["train_dAUC"]  for r in log])

    base_ap, base_au = te_ap[0], te_au[0]          # pretrain-only dZ on test
    oracle_i = int(np.argmax(te_ap))               # best test epoch (not usable)
    proxy_i  = int(np.argmax(tr_ap))               # best train-proxy epoch (usable)
    final_i  = len(log) - 1

    def corr(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    print(f"\n================  FOLD {fold}  ================")
    print(f"  pretrain-only (epoch 0) test dZ : AUPR {base_ap:+.4f} | AUC {base_au:+.4f}")
    print(f"  ---- per-epoch trajectory ----")
    print(f"  {'ep':>3} | {'train_dAUPR':>11} {'train_dAUC':>10} | "
          f"{'test_dAUPR':>10} {'test_dAUC':>9} | {'drift':>6} | "
          f"{'r_mean':>6} {'r_pos':>6} {'r_neg':>6} | {'grad':>5}")
    for r in log:
        print(f"  {r['epoch']:>3} | {r['train_dAUPR']:>+11.4f} {r['train_dAUC']:>+10.4f} | "
              f"{r['test_dAUPR']:>+10.4f} {r['test_dAUC']:>+9.4f} | {r['drift']:>6.3f} | "
              f"{r['reward_mean']:>6.3f} {r['reward_pos']:>6.3f} {r['reward_neg']:>6.3f} | "
              f"{r['grad']:>5.2f}")

    print(f"\n  corr(train_dAUPR, test_dAUPR) = {corr(tr_ap, te_ap):+.3f}   "
          f"corr(train_dAUC, test_dAUC) = {corr(tr_au, te_au):+.3f}")
    print(f"  oracle best test epoch   = {eps[oracle_i]:>3}  -> test dZ-AUPR {te_ap[oracle_i]:+.4f}")
    print(f"  train-proxy best epoch   = {eps[proxy_i]:>3}  -> test dZ-AUPR {te_ap[proxy_i]:+.4f}  (what honest early-stop picks)")
    print(f"  full-run final epoch     = {eps[final_i]:>3}  -> test dZ-AUPR {te_ap[final_i]:+.4f}")
    print(f"  --> early-stop vs full   : {te_ap[proxy_i]-te_ap[final_i]:+.4f} AUPR "
          f"({'helps' if te_ap[proxy_i] > te_ap[final_i] else 'no gain'})")
    print(f"  --> test dZ at stop      : {te_ap[proxy_i]:+.4f} "
          f"(<0 means still below the S+L baseline, just less so)")
    return dict(fold=fold, base_ap=base_ap, oracle=te_ap[oracle_i],
                proxy=te_ap[proxy_i], final=te_ap[final_i],
                corr_ap=corr(tr_ap, te_ap), corr_au=corr(tr_au, te_au))


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--kfold", type=int, default=10)
    pa.add_argument("--folds", type=int, nargs="+", default=[8, 9])
    pa.add_argument("--clf", default="tabpfn", choices=["xgb", "cat", "tabpfn"],
                    help="classifier for dZ eval + train proxy")
    pa.add_argument("--encoder", default="final", choices=["final", "pool"])
    pa.add_argument("--pretrain_epochs", type=int, default=20)
    pa.add_argument("--rl_epochs", type=int, default=60)
    pa.add_argument("--eval_every", type=int, default=2)
    pa.add_argument("--seed", type=int, default=27)
    args = pa.parse_args()

    patients = load_and_prepare_patients()
    feats = get_all_temporal_features(patients)
    enc = SimpleStaticEncoder(FIXED_FEATURES); enc.fit(patients.patientList)
    all_folds = list(trainTestPatients(patients, k=args.kfold, seed=args.seed))

    summ = []
    for fi in args.folds:
        train_full, test_p = all_folds[fi]
        tr_obj, val_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42)
        tp = tr_obj.patientList
        stats = HybridDataset(tp, feats, enc).get_normalization_stats()
        tr_loader = DataLoader(HybridDataset(tp, feats, enc, stats), batch_size=32,
                               shuffle=True, collate_fn=hybrid_collate_fn)
        val_loader = DataLoader(HybridDataset(val_obj.patientList, feats, enc, stats),
                                batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)

        torch.manual_seed(0); np.random.seed(0)
        net = make_net(args.encoder, len(feats)).to(DEVICE)
        net = supervised_pretrain(net, tr_loader, val_loader, args.pretrain_epochs)

        log = rl_finetune_logged(net, tp, test_p.patientList, feats, enc, stats,
                                 reward_clf=args.clf, eval_clf=args.clf,
                                 epochs=args.rl_epochs, eval_every=args.eval_every)
        summ.append(analyse(fi, log))

    print("\n\n================  CROSS-FOLD SUMMARY  ================")
    print(f"  {'fold':>4} | {'base(S+L+Z pre)':>15} | {'oracle':>8} {'proxy-stop':>10} "
          f"{'full':>8} | {'corr_AUPR':>9} {'corr_AUC':>8}")
    for s in summ:
        print(f"  {s['fold']:>4} | {s['base_ap']:>+15.4f} | {s['oracle']:>+8.4f} "
              f"{s['proxy']:>+10.4f} {s['final']:>+8.4f} | "
              f"{s['corr_ap']:>+9.3f} {s['corr_au']:>+8.3f}")
    print("\n  Read:")
    print("  * corr_AUPR > 0 and large  -> train OOF dZ tracks test dZ; early-stop")
    print("    on the train proxy is a valid honest brake for bad folds.")
    print("  * proxy-stop >= full on the bad fold (8) -> stopping caps the damage.")
    print("  * if reward_mean keeps climbing while test_dAUPR falls (see fold 8")
    print("    trajectory), RL is over-optimising a saturating reward, not learning.")


if __name__ == "__main__":
    main()