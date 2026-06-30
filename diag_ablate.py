"""
Ablation: is the learned temporal representation z worth anything, and does it
need last / static?  Measured via OOF-CV inside train (low noise, ~155 pos),
across XGBoost / CatBoost / TabPFN.

Feature blocks: S=static, L=last, MS=mean+std temporal stats, Z=encoder mu.
Configs:
  S, S+L, S+L+MS, S+L+Z, S+L+MS+Z, Z, S+Z

z = deterministic posterior mean, as at inference. Encoder is supervised-
pretrained (no RL) — we are measuring the representation, not the fine-tune.

Run:
  python diag_ablate.py --fold 0 --cv 5 --clf xgb cat        # fast
  python diag_ablate.py --fold 0 --cv 5 --clf xgb cat tabpfn # add TabPFN
"""
import argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from TabPFNRL import (
    FIXED_FEATURES, SupervisedHead, RNNPolicyNetwork,
    HybridDataset, hybrid_collate_fn, SimpleStaticEncoder,
)
from pooled_encoder import PooledRNNPolicyNetwork
from TimeEmbedding import DEVICE
from TimeEmbeddingVal import (
    get_all_temporal_features, split_patients_train_val,
    load_and_prepare_patients,
)
from utils.prepare_data import trainTestPatients


def make_clf(name, ratio):
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=ratio, random_state=42,
                             eval_metric="auc")
    if name == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05,
                                  loss_function="Logloss", eval_metric="AUC",
                                  scale_pos_weight=ratio, random_seed=42,
                                  verbose=False, allow_writing_files=False,
                                  task_type="CPU")
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
    raise ValueError(name)


def pretrain_encoder(make_net, patients, temporal_feats, encoder, stats, epochs, lr=1e-3):
    ds = HybridDataset(patients, temporal_feats, encoder, stats)
    ld = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=hybrid_collate_fn)
    torch.manual_seed(0); np.random.seed(0)
    net = make_net().to(DEVICE)
    head = SupervisedHead(net.latent_dim + len(FIXED_FEATURES)).to(DEVICE)
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=lr)
    crit = nn.BCELoss()
    for _ in range(epochs):
        net.train(); head.train()
        for t_data, lbl, s_data in ld:
            lbl = lbl.to(DEVICE); s_data = s_data.to(DEVICE)
            z, _, _ = net(t_data, deterministic=True)
            preds = head(torch.cat([z, s_data], dim=1)).squeeze(-1)
            loss = crit(preds, lbl)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(net.parameters()) + list(head.parameters()), 1.0)
            opt.step()
    net.eval()
    return net


def extract_blocks(net, patients, temporal_feats, encoder, stats):
    ds = HybridDataset(patients, temporal_feats, encoder, stats)
    ld = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
    S, L, MS, Z, Y = [], [], [], [], []
    with torch.no_grad():
        for t_data, labels, s_data in ld:
            _, _, mean = net(t_data, deterministic=True)
            z_np = mean.cpu().numpy()
            vals = t_data['values'].cpu().numpy()
            masks = t_data['masks'].cpu().numpy()
            for i in range(len(vals)):
                last, mn, sd = [], [], []
                for f in range(vals.shape[2]):
                    v = vals[i, :, f]; m = masks[i, :, f]
                    idx = np.where(m > 0)[0]
                    if len(idx) > 0:
                        vv = v[idx]
                        last.append(vv[-1]); mn.append(np.mean(vv))
                        sd.append(np.std(vv) if len(vv) > 1 else 0.0)
                    else:
                        last.append(0.0); mn.append(0.0); sd.append(0.0)
                L.append(last); MS.append(mn + sd)
            S.append(s_data.numpy()); Z.append(z_np); Y.extend(labels.numpy())
    return {"S": np.vstack(S), "L": np.array(L),
            "MS": np.array(MS), "Z": np.vstack(Z)}, np.array(Y)


def oof_eval(blocks, Y, spec, clf_name, K, seed=0):
    X = np.hstack([blocks[b] for b in spec])
    ratio = float((Y == 0).sum()) / max(int((Y == 1).sum()), 1)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    oof = np.full(len(Y), np.nan)
    for tr, va in skf.split(X, Y):
        clf = make_clf(clf_name, ratio)
        clf.fit(X[tr], Y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
    return average_precision_score(Y, oof), roc_auc_score(Y, oof)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="which outer folds to run (default all 5)")
    pa.add_argument("--cv", type=int, default=5)
    pa.add_argument("--pretrain_epochs", type=int, default=20)
    pa.add_argument("--seed", type=int, default=27)
    pa.add_argument("--clf", nargs="+", default=["xgb", "cat"],
                    choices=["xgb", "cat", "tabpfn"])
    pa.add_argument("--encoder", default="pool", choices=["final", "pool"],
                    help="final = original last-hidden-state RNN; "
                         "pool = statistical-pooling encoder (mean/std/max/last)")
    args = pa.parse_args()

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)
    encoder = SimpleStaticEncoder(FIXED_FEATURES)
    encoder.fit(patients.patientList)
    all_folds = list(trainTestPatients(patients, seed=args.seed))

    def make_net():
        Net = RNNPolicyNetwork if args.encoder == "final" else PooledRNNPolicyNetwork
        return Net(input_dim=len(temporal_feats),
                   hidden_dim=20, latent_dim=28, time_dim=32)

    configs = [
        ("S", ["S"]), ("S+L", ["S", "L"]), ("S+L+MS", ["S", "L", "MS"]),
        ("S+L+Z", ["S", "L", "Z"]), ("S+L+MS+Z", ["S", "L", "MS", "Z"]),
        ("Z", ["Z"]), ("S+Z", ["S", "Z"]),
    ]
    deltas = [
        ("Z over last+static  (S+L+Z)-(S+L)",     "S+L+Z",    "S+L"),
        ("Z beyond hand-stats (S+L+MS+Z)-(S+L+MS)", "S+L+MS+Z", "S+L+MS"),
        ("learned-temporal vs last  Z-(S+L)",     "Z",        "S+L"),
        ("can Z replace last  (S+Z)-(S+L)",       "S+Z",      "S+L"),
        ("hand-stats value   (S+L+MS)-(S+L)",     "S+L+MS",   "S+L"),
    ]

    print(f"[encoder = {args.encoder}] folds={args.folds} cv={args.cv}\n")

    # results[clf][config] = list of (aupr, auc) across folds
    results = {c: {name: [] for name, _ in configs} for c in args.clf}

    for fi in args.folds:
        train_full, _ = all_folds[fi]
        train_p_obj, _ = split_patients_train_val(
            train_full, val_ratio=0.1, seed=42 + fi)
        train_patients = train_p_obj.patientList
        stats = HybridDataset(train_patients, temporal_feats, encoder).get_normalization_stats()
        net = pretrain_encoder(make_net, train_patients, temporal_feats, encoder,
                               stats, args.pretrain_epochs)
        blocks, Y = extract_blocks(net, train_patients, temporal_feats, encoder, stats)
        print(f"fold {fi}: n={len(Y)} pos={int(Y.sum())} ({Y.mean():.1%})", flush=True)
        for clf_name in args.clf:
            for name, spec in configs:
                aupr, auc = oof_eval(blocks, Y, spec, clf_name, args.cv, seed=args.seed)
                results[clf_name][name].append((aupr, auc))

    def ms(vals, i):
        a = np.array([v[i] for v in vals])
        return a.mean(), a.std()

    for clf_name in args.clf:
        R = results[clf_name]
        print(f"\n================  {clf_name.upper()}  "
              f"mean±std over {len(args.folds)} folds  ================")
        print(f"{'config':<12} | {'AUPR mean±std':>17} | {'AUC mean±std':>17}")
        print("-" * 54)
        for name, _ in configs:
            am, asd = ms(R[name], 0); cm, csd = ms(R[name], 1)
            print(f"{name:<12} | {am:.4f} ± {asd:.4f}  | {cm:.4f} ± {csd:.4f}")
        print("  deltas (mean±std; >0 and |mean|>std means it survives fold noise):")
        for label, a, b in deltas:
            da = np.array([x[0] for x in R[a]]) - np.array([x[0] for x in R[b]])
            dc = np.array([x[1] for x in R[a]]) - np.array([x[1] for x in R[b]])
            flag_a = "OK" if abs(da.mean()) > da.std() else "noise"
            flag_c = "OK" if abs(dc.mean()) > dc.std() else "noise"
            print(f"   {label:<40}: "
                  f"AUPR {da.mean():+.4f}±{da.std():.4f}[{flag_a}]  "
                  f"AUC {dc.mean():+.4f}±{dc.std():.4f}[{flag_c}]")


if __name__ == "__main__":
    main()