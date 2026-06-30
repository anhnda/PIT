"""
Does RL fine-tuning add anything ON TOP of the pretrained pooled encoder?

Compares, with identical OOF-CV evaluation (TabPFN/XGB/CatBoost), per fold:
   pooled + supervised-pretrain        (the z we already measured)
   pooled + supervised-pretrain + RL   (REINFORCE fine-tune)

The RL loop here fixes the three issues that made the original RL inert:
  1. OUT-OF-FOLD reward: reward for each train patient comes from a classifier
     fit on OTHER train patients (StratifiedKFold inside train), not from
     predict_proba on the same in-context set (which saturates -> zero-signal
     reward -> flat val). This is internal cross-fitting; outer val/test are
     untouched.
  2. Proper entropy bonus (+H, encourages exploration) instead of the original
     `- 0.001*log_prob` which pushed the policy to collapse.
  3. No hard temperature floor; std is governed by the entropy term.
  Reward is per-sample train quality only (no val_aupr leak).

We measure z BEFORE and AFTER RL via the same delta the ablation used:
   Z over last+static = (S+L+Z) - (S+L), OOF, mean±std over folds.
If RL's delta > supervised delta and survives fold noise, RL earns its place.
If not, the contribution is the pooling encoder, not the policy gradient.

Run:  python diag_rl.py --clf xgb --folds 0 1 2 3 4 --rl_epochs 40
"""
import argparse, copy, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from TabPFNRL import (
    FIXED_FEATURES, SupervisedHead, HybridDataset, hybrid_collate_fn,
    SimpleStaticEncoder,
)
from pooled_encoder import PooledRNNPolicyNetwork
from TimeEmbedding import DEVICE
from TimeEmbeddingVal import (
    get_all_temporal_features, split_patients_train_val, load_and_prepare_patients,
)
from utils.prepare_data import trainTestPatients


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


def make_net(n_feat):
    return PooledRNNPolicyNetwork(input_dim=n_feat, hidden_dim=20,
                                  latent_dim=28, time_dim=32)


def supervised_pretrain(net, loader, epochs, lr=1e-3):
    head = SupervisedHead(net.latent_dim + len(FIXED_FEATURES)).to(DEVICE)
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=lr)
    crit = nn.BCELoss()
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
    net.eval(); return net


def collect_z_static_label(net, patients, feats, enc, stats, sample=False, temp=1.0):
    ds = HybridDataset(patients, feats, enc, stats)
    ld = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
    Z, S, Y, LP = [], [], [], []
    for t, lbl, s in ld:
        z, logp, mean = net(t, deterministic=not sample, temperature=temp)
        Z.append((z if sample else mean).detach().cpu().numpy())
        S.append(s.numpy()); Y.extend(lbl.numpy())
        LP.append(logp.detach().cpu().numpy() if (sample and logp is not None) else None)
    return np.vstack(Z), np.vstack(S), np.array(Y)


def oof_reward(Xz_static, Y, clf_name, K=5, seed=0):
    """Out-of-fold per-sample reward: prob assigned to the TRUE class by a
    classifier that did NOT see this sample (fit on other inner folds)."""
    ratio = float((Y == 0).sum()) / max(int((Y == 1).sum()), 1)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    proba = np.full(len(Y), np.nan)
    for tr, va in skf.split(Xz_static, Y):
        clf = make_clf(clf_name, ratio)
        clf.fit(Xz_static[tr], Y[tr])
        proba[va] = clf.predict_proba(Xz_static[va])[:, 1]
    reward = np.where(Y == 1, proba, 1.0 - proba)
    return reward


def rl_finetune(net, patients, feats, enc, stats, clf_name, epochs, lr=3e-4,
                ent_coef=0.01):
    """REINFORCE fine-tune with out-of-fold reward and a proper entropy bonus."""
    ds = HybridDataset(patients, feats, enc, stats)
    ld = DataLoader(ds, batch_size=len(patients), shuffle=False,
                    collate_fn=hybrid_collate_fn)  # full-batch for simple REINFORCE
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # static + label are fixed; precompute static per patient via one pass
    for epoch in range(epochs):
        net.train()
        for t, lbl, s in ld:
            s = s.to(DEVICE)
            # sample z, get log_prob and the gaussian entropy
            z, logp, mean = net(t, deterministic=False, temperature=1.0)
            # build [static || z] for the reward model (last/MS handled outside;
            # here reward is on static+z which is enough to shape z)
            Xz = torch.cat([s, z], dim=1).detach().cpu().numpy()
            Y = lbl.numpy().astype(int)
            reward = oof_reward(Xz, Y, clf_name)          # out-of-fold, real signal
            R = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
            R = (R - R.mean()) / (R.std() + 1e-8)
            # gaussian entropy of the diagonal policy (encourage exploration)
            # entropy of N(mu, sigma) = 0.5*log(2*pi*e*sigma^2) summed over dims
            # recover sigma from logp is awkward; instead recompute via net stats:
            # use a small proxy: penalize collapse by maximizing sample spread
            policy_loss = -(logp * R).mean()
            ent = 0.5 * torch.log(2 * np.pi * np.e * (z.var(dim=0) + 1e-6)).sum()
            loss = policy_loss - ent_coef * ent
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    net.eval(); return net


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


def oof_eval(blocks, Y, spec, clf_name, K=5, seed=0):
    X = np.hstack([blocks[b] for b in spec])
    ratio = float((Y == 0).sum()) / max(int((Y == 1).sum()), 1)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    oof = np.full(len(Y), np.nan)
    for tr, va in skf.split(X, Y):
        clf = make_clf(clf_name, ratio)
        clf.fit(X[tr], Y[tr]); oof[va] = clf.predict_proba(X[va])[:, 1]
    return average_precision_score(Y, oof), roc_auc_score(Y, oof)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    pa.add_argument("--clf", default="xgb", choices=["xgb", "cat", "tabpfn"])
    pa.add_argument("--pretrain_epochs", type=int, default=20)
    pa.add_argument("--rl_epochs", type=int, default=40)
    pa.add_argument("--seed", type=int, default=27)
    args = pa.parse_args()

    patients = load_and_prepare_patients()
    feats = get_all_temporal_features(patients)
    enc = SimpleStaticEncoder(FIXED_FEATURES); enc.fit(patients.patientList)
    all_folds = list(trainTestPatients(patients, seed=args.seed))

    d_pre = {"aupr": [], "auc": []}
    d_rl = {"aupr": [], "auc": []}

    for fi in args.folds:
        train_full, _ = all_folds[fi]
        tr_obj, _ = split_patients_train_val(train_full, val_ratio=0.1, seed=42 + fi)
        tp = tr_obj.patientList
        stats = HybridDataset(tp, feats, enc).get_normalization_stats()
        loader = DataLoader(HybridDataset(tp, feats, enc, stats), batch_size=32,
                            shuffle=True, collate_fn=hybrid_collate_fn)

        # 1) pooled + supervised
        net = make_net(len(feats)).to(DEVICE)
        net = supervised_pretrain(net, loader, args.pretrain_epochs)
        b, Y = blocks_for_eval(net, tp, feats, enc, stats)
        a0, c0 = oof_eval(b, Y, ["S", "L"], args.clf, seed=args.seed)
        a1, c1 = oof_eval(b, Y, ["S", "L", "Z"], args.clf, seed=args.seed)
        d_pre["aupr"].append(a1 - a0); d_pre["auc"].append(c1 - c0)

        # 2) + RL fine-tune (continue from the same pretrained net)
        net = rl_finetune(net, tp, feats, enc, stats, args.clf, args.rl_epochs)
        b2, Y2 = blocks_for_eval(net, tp, feats, enc, stats)
        a0b, c0b = oof_eval(b2, Y2, ["S", "L"], args.clf, seed=args.seed)
        a1b, c1b = oof_eval(b2, Y2, ["S", "L", "Z"], args.clf, seed=args.seed)
        d_rl["aupr"].append(a1b - a0b); d_rl["auc"].append(c1b - c0b)

        print(f"fold {fi}: pretrain dZ AUPR {a1-a0:+.4f} AUC {c1-c0:+.4f} | "
              f"+RL dZ AUPR {a1b-a0b:+.4f} AUC {c1b-c0b:+.4f}", flush=True)

    def ms(x): a = np.array(x); return a.mean(), a.std()
    print(f"\n=== Z-over-(S+L), clf={args.clf}, {len(args.folds)} folds ===")
    for tag, d in [("pooled+supervised", d_pre), ("pooled+supervised+RL", d_rl)]:
        am, asd = ms(d["aupr"]); cm, csd = ms(d["auc"])
        fa = "OK" if abs(am) > asd else "noise"
        fc = "OK" if abs(cm) > csd else "noise"
        print(f"  {tag:22s} | AUPR {am:+.4f}±{asd:.4f}[{fa}] | AUC {cm:+.4f}±{csd:.4f}[{fc}]")
    da = np.array(d_rl["aupr"]) - np.array(d_pre["aupr"])
    dc = np.array(d_rl["auc"]) - np.array(d_pre["auc"])
    print(f"\n  RL contribution beyond pretrain: "
          f"AUPR {da.mean():+.4f}±{da.std():.4f}[{'OK' if abs(da.mean())>da.std() else 'noise'}] "
          f"AUC {dc.mean():+.4f}±{dc.std():.4f}[{'OK' if abs(dc.mean())>dc.std() else 'noise'}]")
    print("  If this is ~0/noise, the pooling encoder is the contribution, not RL.")


if __name__ == "__main__":
    main()