"""
diag_rl_holdout.py  --  RL evaluation on a FIXED, RANDOM hold-out test set.

Difference from diag_rl.py (which did rotating k-fold CV, every patient tested
once): here a single hold-out test partition is carved out ONCE, by a fixed
seed, BEFORE anything else, and is NEVER touched during training, val-selection,
RL reward, or model choice. It enters only at the final scoring call.

The remaining (non-hold-out) patients are run through k-fold CV to produce k
INDEPENDENT encoders per type (pretrain / +RL / scratch). Each of the k encoders
is trained only on its CV-train partition (a val slice carved from it picks the
supervised checkpoint) and then scored on the SAME frozen hold-out. This gives:

  * a clean estimate of dZ on a test set the rule/encoder never saw, AND
  * k replicates whose spread comes from training variability (CV split + init),
    NOT from test-split variability -> mean±std and paired tests across the k
    replicates are valid (all replicates share the identical hold-out, so
    pre/+RL/scratch are paired replicate-by-replicate).

dZ = (S+L+Z) - (S+L) on the frozen hold-out, AUPR and AUC. Absolute (S+L) and
(S+L+Z) are also reported so a negative dZ can be read as "S+L high" vs
"S+L+Z low" rather than guessed.

Run:  python diag_rl_holdout.py --clf tabpfn --encoder final \
          --holdout_frac 0.2 --kfold 5 --rl_epochs 40 --holdout_seed 12345
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
from utils.class_patient import Patients


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
    ds = HybridDataset(patients, feats, enc, stats)
    ld = DataLoader(ds, batch_size=len(patients), shuffle=False,
                    collate_fn=hybrid_collate_fn)
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


# ---------- blocks: S, L (last-observed), Z ----------
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


def dZ_on_test(net, train_p, test_p, feats, enc, stats, clf):
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


# ---------- fixed random hold-out, stratified ----------
def carve_holdout(patients, holdout_frac, seed):
    """Stratified split into (remainder, holdout). holdout_frac of patients go to
    the frozen test set. Uses Patients.split with n = round(1/frac) buckets and
    takes bucket 0 as holdout -> deterministic in `seed`, never touched again."""
    n_buckets = max(2, round(1.0 / holdout_frac))
    buckets = patients.split(n_buckets, seed)          # stratified, cached by seed
    holdout = buckets[0]
    remainder = Patients(patients=[])
    for b in buckets[1:]:
        remainder = remainder + b
    return remainder, holdout, n_buckets


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--holdout_frac", type=float, default=0.2,
                    help="fraction of patients frozen as the hold-out test set")
    pa.add_argument("--holdout_seed", type=int, default=12345,
                    help="seed for the FIXED hold-out carve (fix once, never tune)")
    pa.add_argument("--kfold", type=int, default=5,
                    help="CV folds on the REMAINDER -> k independent encoders")
    pa.add_argument("--cv_seed", type=int, default=27,
                    help="seed for the CV split of the remainder")
    pa.add_argument("--clf", default="tabpfn", choices=["xgb", "cat", "tabpfn"])
    pa.add_argument("--encoder", default="final", choices=["final", "pool"])
    pa.add_argument("--pretrain_epochs", type=int, default=20)
    pa.add_argument("--rl_epochs", type=int, default=40)
    args = pa.parse_args()

    patients = load_and_prepare_patients()
    feats = get_all_temporal_features(patients)
    enc = SimpleStaticEncoder(FIXED_FEATURES); enc.fit(patients.patientList)

    remainder, holdout, nb = carve_holdout(patients, args.holdout_frac, args.holdout_seed)
    holdout_p = holdout.patientList
    h_pos = sum(int(p.akdPositive) for p in holdout_p)
    print(f"[holdout] frac~1/{nb}  seed={args.holdout_seed}  "
          f"N_holdout={len(holdout_p)} (pos {h_pos}, rate {h_pos/max(len(holdout_p),1):.3f}) "
          f"| N_remainder={len(remainder)}  FROZEN, never seen in training", flush=True)

    # CV on the remainder only -> k independent train partitions
    cv_buckets = remainder.split(args.kfold, args.cv_seed)

    rec = {k: {"aupr": [], "auc": [], "full_ap": [], "full_au": [],
               "base_ap": [], "base_au": []}
           for k in ["pre", "rl", "scratch"]}

    for fi in range(args.kfold):
        # CV-train = remainder minus bucket fi (hold-out is already outside)
        cv_train = Patients(patients=[])
        for j, b in enumerate(cv_buckets):
            if j != fi:
                cv_train = cv_train + b
        tr_obj, val_obj = split_patients_train_val(cv_train, val_ratio=0.1, seed=42)
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

        # 1) supervised pretrain -> score on FROZEN hold-out
        torch.manual_seed(0); np.random.seed(0)
        net = make_net(args.encoder, len(feats)).to(DEVICE)
        net = supervised_pretrain(net, tr_loader, val_loader, args.pretrain_epochs)
        dZ, full, base = dZ_on_test(net, tp, holdout_p, feats, enc, stats, args.clf)
        store("pre", dZ, full, base)

        # 2) + RL fine-tune (continue from pretrained) -> hold-out
        net = rl_finetune(net, tp, feats, enc, stats, args.clf, args.rl_epochs)
        dZ2, full2, base2 = dZ_on_test(net, tp, holdout_p, feats, enc, stats, args.clf)
        store("rl", dZ2, full2, base2)

        # 3) RL from scratch (random init, matched budget) -> hold-out
        torch.manual_seed(0); np.random.seed(0)
        net_s = make_net(args.encoder, len(feats)).to(DEVICE)
        net_s = rl_finetune(net_s, tp, feats, enc, stats, args.clf,
                            args.rl_epochs + args.pretrain_epochs)
        dZ3, full3, base3 = dZ_on_test(net_s, tp, holdout_p, feats, enc, stats, args.clf)
        store("scratch", dZ3, full3, base3)

        print(f"cv-rep {fi}: dZ-AUPR  pre {dZ[0]:+.4f} | +RL {dZ2[0]:+.4f} | "
              f"scratch {dZ3[0]:+.4f}   ||   dZ-AUC  pre {dZ[1]:+.4f} | "
              f"+RL {dZ2[1]:+.4f} | scratch {dZ3[1]:+.4f}", flush=True)

    def ms(x): a = np.array(x); return a.mean(), a.std()
    tags = [("pre", "supervised-pretrain"),
            ("rl", "supervised + RL"),
            ("scratch", "RL-from-scratch")]

    print(f"\n=== FROZEN HOLD-OUT, clf={args.clf}, encoder={args.encoder}, "
          f"{args.kfold} CV-replicates on remainder ===")
    print(f"    (all replicates scored on the SAME hold-out; spread = training "
          f"variability, not test-split)")

    print(f"\n  dZ = (S+L+Z) - (S+L) on hold-out, mean±std over {args.kfold} replicates")
    for key, label in tags:
        am, asd = ms(rec[key]["aupr"]); cm, csd = ms(rec[key]["auc"])
        print(f"    {label:24s} | AUPR dZ {am:+.4f}±{asd:.4f} | "
              f"AUC dZ {cm:+.4f}±{csd:.4f}")

    print(f"\n  absolute on hold-out, mean±std  (read negative dZ here: S+L high vs S+L+Z low)")
    for key, label in tags:
        fam, fasd = ms(rec[key]["full_ap"]); fcm, fcsd = ms(rec[key]["full_au"])
        bam, basd = ms(rec[key]["base_ap"]); bcm, bcsd = ms(rec[key]["base_au"])
        print(f"    {label:24s} | (S+L+Z) AUPR {fam:.4f}±{fasd:.4f} AUC {fcm:.4f}±{fcsd:.4f}"
              f"  |  (S+L) AUPR {bam:.4f}±{basd:.4f} AUC {bcm:.4f}±{bcsd:.4f}")

    print(f"\n  significance: S+L+Z vs S+L (paired across {args.kfold} replicates)")
    for key, label in tags:
        print(f"   [{label}]")
        paired_report("AUC-ROC (S+L+Z) vs (S+L)", rec[key]["full_au"], rec[key]["base_au"])
        paired_report("AUPR    (S+L+Z) vs (S+L)", rec[key]["full_ap"], rec[key]["base_ap"])

    print(f"\n  significance: does RL beat supervised? (paired dZ across replicates)")
    paired_report("AUC-ROC (+RL dZ) vs (pretrain dZ)", rec["rl"]["auc"], rec["pre"]["auc"])
    paired_report("AUPR    (+RL dZ) vs (pretrain dZ)", rec["rl"]["aupr"], rec["pre"]["aupr"])
    paired_report("AUC-ROC (scratch dZ) vs (pretrain dZ)", rec["scratch"]["auc"], rec["pre"]["auc"])
    paired_report("AUPR    (scratch dZ) vs (pretrain dZ)", rec["scratch"]["aupr"], rec["pre"]["aupr"])
    print("\n  Read: hold-out is frozen by --holdout_seed and never enters training.")
    print("  Replicate spread is training noise (CV split + init), so pre/+RL/scratch")
    print("  are paired replicate-by-replicate on the identical test set.")


if __name__ == "__main__":
    main()