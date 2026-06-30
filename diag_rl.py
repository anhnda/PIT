"""
Does RL (REINFORCE) earn its place, evaluated HONESTLY on held-out test?

Leak-free counterpart of the earlier diag_rl. Every encoder (supervised-
pretrained, +RL fine-tuned, RL-from-scratch) is trained ONLY on the fold's
train partition; a val split carved from train selects the supervised
checkpoint; z for the TEST patients comes from an encoder that never saw them.
A classifier is fit on train-[features] and scored on the held-out TEST. This
matches diag_test.py exactly.

It replaces the previous version, which did OOF *inside train* (encoder
pretrained on all of train, then z measured via inner CV on the same patients):
the label leaked through the encoder and the test set was never touched. Those
numbers were optimistic.

Three encoders, identical protocol, per fold:
   <enc> + supervised-pretrain            baseline z (what diag_test measured)
   <enc> + supervised-pretrain + RL       REINFORCE fine-tune on top
   <enc> + RL-from-scratch                random init, RL only, matched budget

RL loop fixes that keep REINFORCE from being inert:
  1. OUT-OF-FOLD reward: each train patient's reward = prob the true class gets
     from a classifier fit on OTHER train patients (inner StratifiedKFold).
     Avoids predict_proba-on-self saturation. Outer val/test untouched.
  2. Real Gaussian entropy bonus (+H) instead of the original -0.001*log_prob
     that collapsed the policy.
  3. No hard temperature floor; std governed by the entropy term.

Reports per encoder: dZ = (S+L+Z)-(S+L) on held-out TEST (AUPR and AUC),
mean+-std over folds; a paired test (paired-t + Wilcoxon + Cohen dz) of S+L+Z
vs S+L; and paired tests of (+RL) vs (supervised) and (RL-scratch) vs
(supervised). RL is worth keeping only if its dZ beats supervised pretrain AND
survives fold noise. If RL-from-scratch is well below supervised, RL cannot
even match a plain BCE encoder on this task.

Run:  python diag_rl.py --clf tabpfn --encoder final --kfold 10 --rl_epochs 40
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


# ---------- supervised pretrain (train-only, val checkpoint) ----------
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


# ---------- RL fine-tune (out-of-fold reward, train-only) ----------
def oof_reward(Xz_static, Y, clf_name, K=5, seed=0):
    """Per-sample reward = prob the TRUE class gets from a classifier that did
    NOT see this sample (inner StratifiedKFold over TRAIN only)."""
    ratio = float((Y == 0).sum()) / max(int((Y == 1).sum()), 1)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    proba = np.full(len(Y), np.nan)
    for tr, va in skf.split(Xz_static, Y):
        clf = make_clf(clf_name, ratio)
        clf.fit(Xz_static[tr], Y[tr])
        proba[va] = clf.predict_proba(Xz_static[va])[:, 1]
    return np.where(Y == 1, proba, 1.0 - proba)


def rl_finetune(net, patients, feats, enc, stats, clf_name, epochs,
                lr=3e-4, ent_coef=0.01):
    """REINFORCE fine-tune with out-of-fold reward + entropy bonus. Train-only:
    reward classifiers fit on inner folds of TRAIN; test never enters. Final
    encoder state is returned (RL has no separate head to overfit)."""
    ds = HybridDataset(patients, feats, enc, stats)
    ld = DataLoader(ds, batch_size=len(patients), shuffle=False,
                    collate_fn=hybrid_collate_fn)  # full-batch REINFORCE
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        net.train()
        for t, lbl, s in ld:
            s = s.to(DEVICE)
            z, logp, mean = net(t, deterministic=False, temperature=1.0)
            Xz = torch.cat([s, z], dim=1).detach().cpu().numpy()
            Y = lbl.numpy().astype(int)
            reward = oof_reward(Xz, Y, clf_name)
            R = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
            R = (R - R.mean()) / (R.std() + 1e-8)
            policy_loss = -(logp * R).mean()
            ent = 0.5 * torch.log(2 * np.pi * np.e * (z.var(dim=0) + 1e-6)).sum()
            loss = policy_loss - ent_coef * ent
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    net.eval(); return net


# ---------- blocks: S, L (last-observed), Z, on any patient set ----------
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
    """Fit on TRAIN blocks, score on held-out TEST blocks."""
    Xtr = np.hstack([tr_b[k] for k in spec]); Xte = np.hstack([te_b[k] for k in spec])
    ratio = float((tr_Y == 0).sum()) / max(int((tr_Y == 1).sum()), 1)
    c = make_clf(clf_name, ratio); c.fit(Xtr, tr_Y)
    p = c.predict_proba(Xte)[:, 1]
    return average_precision_score(te_Y, p), roc_auc_score(te_Y, p)


def dZ_on_test(net, train_p, test_p, feats, enc, stats, clf):
    """dZ = (S+L+Z) - (S+L) on held-out test; returns (dZ, full, base) for
    both AUPR and AUC."""
    tr_b, tr_Y = blocks_for_eval(net, train_p, feats, enc, stats)
    te_b, te_Y = blocks_for_eval(net, test_p, feats, enc, stats)
    a0, c0 = fit_score(tr_b, tr_Y, te_b, te_Y, ["S", "L"], clf)
    a1, c1 = fit_score(tr_b, tr_Y, te_b, te_Y, ["S", "L", "Z"], clf)
    return (a1 - a0, c1 - c0), (a1, c1), (a0, c0)


def paired_report(name, a, b):
    a = np.array(a); b = np.array(b); d = a - b; n = len(d)
    t_stat, t_p = sp_stats.ttest_rel(a, b)
    try:
        w_stat, w_p = sp_stats.wilcoxon(a, b); w_str = f"W={w_stat:.1f}, p={w_p:.4f}"
    except ValueError as e:
        w_str = f"n/a ({e})"
    dz = d.mean() / (d.std(ddof=1) + 1e-12)
    wins = int((d > 0).sum())
    print(f"    {name:36s} mean={d.mean():+.4f} wins {wins}/{n} | "
          f"paired-t t={t_stat:+.3f} p={t_p:.4f} | Wilcoxon {w_str} | dz={dz:+.2f}")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--kfold", type=int, default=10, help="number of outer CV folds")
    pa.add_argument("--folds", type=int, nargs="+", default=None,
                    help="which fold indices to run (default: all 0..kfold-1)")
    pa.add_argument("--clf", default="tabpfn", choices=["xgb", "cat", "tabpfn"])
    pa.add_argument("--encoder", default="final", choices=["final", "pool"])
    pa.add_argument("--pretrain_epochs", type=int, default=20)
    pa.add_argument("--rl_epochs", type=int, default=40)
    pa.add_argument("--seed", type=int, default=27)
    args = pa.parse_args()
    if args.folds is None:
        args.folds = list(range(args.kfold))

    patients = load_and_prepare_patients()
    feats = get_all_temporal_features(patients)
    enc = SimpleStaticEncoder(FIXED_FEATURES); enc.fit(patients.patientList)
    all_folds = list(trainTestPatients(patients, k=args.kfold, seed=args.seed))

    rec = {k: {"aupr": [], "auc": [], "full_ap": [], "full_au": [],
               "base_ap": [], "base_au": []}
           for k in ["pre", "rl", "scratch"]}

    for fi in args.folds:
        train_full, test_p = all_folds[fi]
        tr_obj, val_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42)
        tp = tr_obj.patientList
        stats = HybridDataset(tp, feats, enc).get_normalization_stats()
        tr_loader = DataLoader(HybridDataset(tp, feats, enc, stats), batch_size=32,
                               shuffle=True, collate_fn=hybrid_collate_fn)
        val_loader = DataLoader(HybridDataset(val_obj.patientList, feats, enc, stats),
                                batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)

        def store(key, dZ, full, base):
            rec[key]["aupr"].append(dZ[0]); rec[key]["auc"].append(dZ[1])
            rec[key]["full_ap"].append(full[0]); rec[key]["full_au"].append(full[1])
            rec[key]["base_ap"].append(base[0]); rec[key]["base_au"].append(base[1])

        # 1) supervised pretrain (train-only, val checkpoint)
        torch.manual_seed(0); np.random.seed(0)
        net = make_net(args.encoder, len(feats)).to(DEVICE)
        net = supervised_pretrain(net, tr_loader, val_loader, args.pretrain_epochs)
        dZ, full, base = dZ_on_test(net, tp, test_p.patientList, feats, enc, stats, args.clf)
        store("pre", dZ, full, base)

        # 2) + RL fine-tune (continue from same pretrained encoder; train-only)
        net = rl_finetune(net, tp, feats, enc, stats, args.clf, args.rl_epochs)
        dZ2, full2, base2 = dZ_on_test(net, tp, test_p.patientList, feats, enc, stats, args.clf)
        store("rl", dZ2, full2, base2)

        # 3) RL from scratch (random init, RL only, matched total budget)
        torch.manual_seed(0); np.random.seed(0)
        net_s = make_net(args.encoder, len(feats)).to(DEVICE)
        net_s = rl_finetune(net_s, tp, feats, enc, stats, args.clf,
                            args.rl_epochs + args.pretrain_epochs)
        dZ3, full3, base3 = dZ_on_test(net_s, tp, test_p.patientList, feats, enc, stats, args.clf)
        store("scratch", dZ3, full3, base3)

        print(f"fold {fi}: dZ-AUPR  pre {dZ[0]:+.4f} | +RL {dZ2[0]:+.4f} | "
              f"scratch {dZ3[0]:+.4f}   ||   dZ-AUC  pre {dZ[1]:+.4f} | "
              f"+RL {dZ2[1]:+.4f} | scratch {dZ3[1]:+.4f}", flush=True)

    def ms(x): a = np.array(x); return a.mean(), a.std()
    tags = [("pre", "supervised-pretrain"),
            ("rl", "supervised + RL"),
            ("scratch", "RL-from-scratch")]

    print(f"\n=== HELD-OUT TEST, clf={args.clf}, encoder={args.encoder}, "
          f"{len(args.folds)} folds ===")

    print(f"\n  dZ = (S+L+Z) - (S+L), held-out test, mean±std over folds")
    for key, label in tags:
        am, asd = ms(rec[key]["aupr"]); cm, csd = ms(rec[key]["auc"])
        print(f"    {label:24s} | AUPR dZ {am:+.4f}±{asd:.4f} | "
              f"AUC dZ {cm:+.4f}±{csd:.4f}")

    print(f"\n  absolute (S+L+Z), held-out test, mean±std")
    for key, label in tags:
        am, asd = ms(rec[key]["full_ap"]); cm, csd = ms(rec[key]["full_au"])
        print(f"    {label:24s} | AUPR {am:.4f}±{asd:.4f} | AUC {cm:.4f}±{csd:.4f}")

    print(f"\n  significance: S+L+Z vs S+L (paired across {len(args.folds)} folds)")
    for key, label in tags:
        print(f"   [{label}]")
        paired_report("AUC-ROC (S+L+Z) vs (S+L)",
                      rec[key]["full_au"], rec[key]["base_au"])
        paired_report("AUPR    (S+L+Z) vs (S+L)",
                      rec[key]["full_ap"], rec[key]["base_ap"])

    print(f"\n  significance: does RL beat supervised? (paired dZ across folds)")
    paired_report("AUC-ROC (+RL dZ) vs (pretrain dZ)", rec["rl"]["auc"], rec["pre"]["auc"])
    paired_report("AUPR    (+RL dZ) vs (pretrain dZ)", rec["rl"]["aupr"], rec["pre"]["aupr"])
    paired_report("AUC-ROC (scratch dZ) vs (pretrain dZ)", rec["scratch"]["auc"], rec["pre"]["auc"])
    paired_report("AUPR    (scratch dZ) vs (pretrain dZ)", rec["scratch"]["aupr"], rec["pre"]["aupr"])
    print("\n  Read: RL earns its place only if +RL dZ > pretrain dZ with a real")
    print("  paired effect, AND RL-from-scratch at least matches supervised.")


if __name__ == "__main__":
    main()