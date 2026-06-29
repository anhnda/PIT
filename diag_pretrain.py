"""
Pretrain diagnostic — does NOT modify the pipeline.

Re-creates the EXACT fold setup from TabPFNRL.main() (same dataset, split,
collate, encoder) and re-runs the same pretrain loop while sweeping LR, so you
can see which cause is real:

  (A) LR too high  -> val AUPR peaks in the first few epochs then declines
                      monotonically (overfit). Lowering LR should fix it.
  (B) val too small -> AUPR is just noisy because n_val_pos is tiny
                      (post-leak-fix pos_rate ~21.8%, so ~20-40 pos per fold).
  (C) optim instability -> val swings wildly even at low LR.

Every epoch prints:  train_loss | val_AUPR | val_AUC | n_val_pos / n_val
The script interprets nothing beyond arithmetic. You read the curves.

Run:
    python diag_pretrain.py --fold 0 --lrs 1e-3 1e-4 3e-5 --epochs 60
    python diag_pretrain.py --fold 2 --lrs 1e-3 1e-4 --wd 1e-4
"""
import argparse, copy, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from TabPFNRL import (
    FIXED_FEATURES, SupervisedHead, RNNPolicyNetwork,
    HybridDataset, hybrid_collate_fn, SimpleStaticEncoder,
)
from TimeEmbedding import DEVICE
from TimeEmbeddingVal import (
    get_all_temporal_features, split_patients_train_val,
    load_and_prepare_patients,
)
from utils.prepare_data import trainTestPatients


def count_pos(loader):
    n_pos = n = 0
    for _, labels, _ in loader:
        y = labels.numpy(); n_pos += int(y.sum()); n += len(y)
    return n_pos, n


def run_one(make_net, train_loader, val_loader, lr, epochs, wd, eval_every=1):
    torch.manual_seed(0); np.random.seed(0)
    policy_net = make_net().to(DEVICE)
    head = SupervisedHead(policy_net.latent_dim + len(FIXED_FEATURES)).to(DEVICE)
    opt = torch.optim.Adam(
        list(policy_net.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=wd)
    criterion = nn.BCELoss()
    hist = []
    for epoch in range(epochs):
        policy_net.train(); head.train()
        tot = nb = 0
        for t_data, labels, s_data in train_loader:
            labels = labels.to(DEVICE); s_data = s_data.to(DEVICE)
            z, _, _ = policy_net(t_data, deterministic=True)
            preds = head(torch.cat([z, s_data], dim=1)).squeeze(-1)
            loss = criterion(preds, labels)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy_net.parameters()) + list(head.parameters()), 1.0)
            opt.step(); tot += loss.item(); nb += 1
        if (epoch + 1) % eval_every == 0:
            policy_net.eval(); head.eval()
            ap, al = [], []
            with torch.no_grad():
                for t_data, labels, s_data in val_loader:
                    s_data = s_data.to(DEVICE)
                    z, _, _ = policy_net(t_data, deterministic=True)
                    preds = head(torch.cat([z, s_data], dim=1)).squeeze(-1)
                    ap.extend(preds.cpu().numpy()); al.extend(labels.numpy())
            al = np.array(al); ap = np.array(ap)
            aupr = average_precision_score(al, ap)
            try: auc = roc_auc_score(al, ap)
            except ValueError: auc = float('nan')
            hist.append((epoch + 1, tot / max(nb, 1), aupr, auc, int(al.sum()), len(al)))
    return hist


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--fold", type=int, default=0)
    pa.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 1e-4, 3e-5])
    pa.add_argument("--epochs", type=int, default=60)
    pa.add_argument("--wd", type=float, default=0.0)
    pa.add_argument("--seed", type=int, default=27)
    pa.add_argument("--val_ratio", type=float, default=0.1,
                    help="bump to 0.2/0.3 to see if a larger val tames AUPR noise (cause B)")
    pa.add_argument("--cv", type=int, default=0,
                    help="if >0, also evaluate pretrain via K-fold CV INSIDE train "
                         "(out-of-fold AUPR on all train positives). Decides B vs C.")
    args = pa.parse_args()

    patients = load_and_prepare_patients()
    temporal_feats = get_all_temporal_features(patients)
    encoder = SimpleStaticEncoder(FIXED_FEATURES)
    encoder.fit(patients.patientList)

    folds = list(trainTestPatients(patients, seed=args.seed))
    train_full, test_p = folds[args.fold]
    train_p_obj, val_p_obj = split_patients_train_val(
        train_full, val_ratio=args.val_ratio, seed=42 + args.fold)

    train_ds = HybridDataset(train_p_obj.patientList, temporal_feats, encoder)
    stats = train_ds.get_normalization_stats()
    val_ds = HybridDataset(val_p_obj.patientList, temporal_feats, encoder, stats)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=hybrid_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)

    def make_net():
        return RNNPolicyNetwork(input_dim=len(temporal_feats),
                                hidden_dim=20, latent_dim=28, time_dim=32)

    vpos, vn = count_pos(val_loader); tpos, tn = count_pos(train_loader)
    print(f"\nFOLD {args.fold} | val_ratio={args.val_ratio} | "
          f"train n={tn} pos={tpos} ({tpos/tn:.1%}) "
          f"| val n={vn} pos={vpos} ({vpos/vn:.1%})")
    print(f"NOTE: val has {vpos} positives -> AUPR is intrinsically noisy "
          f"if this is < ~40 (cause B).\n")

    for lr in args.lrs:
        print(f"================  LR={lr:g}  wd={args.wd:g}  val_ratio={args.val_ratio}  ================")
        print(f"{'ep':>3} | {'tr_loss':>8} | {'val_AUPR':>8} | {'val_AUC':>7} | pos/n")
        hist = run_one(make_net, train_loader, val_loader, lr, args.epochs, args.wd)
        for ep, tl, aupr, auc, vp, vnn in hist:
            print(f"{ep:3d} | {tl:8.4f} | {aupr:8.4f} | {auc:7.4f} | {vp}/{vnn}")
        auprs = [h[2] for h in hist]; aucs = [h[3] for h in hist]
        peak_ep = hist[int(np.argmax(auprs))][0]
        last5 = float(np.mean(auprs[-5:])) if len(auprs) >= 5 else float(np.mean(auprs))
        print(f"  -> AUPR peak {max(auprs):.4f}@ep{peak_ep} | mean(last5)={last5:.4f} "
              f"| AUPR swing={max(auprs)-min(auprs):.4f} | AUC swing={max(aucs)-min(aucs):.4f}")
        print(f"     If AUPR swing >> AUC swing, the wobble is metric noise on few "
              f"positives (B), not the model (C).\n")

    if args.cv > 0:
        print(f"\n############  INTERNAL {args.cv}-FOLD CV ON TRAIN  ############")
        print("Out-of-fold AUPR is computed on ALL train positives at once, so it")
        print("has many more positives than a single small val split. If this curve")
        print("is stable while the val curves above wobble, the cause is B (small")
        print("val), not C (unstable training). val/test stay untouched.\n")
        run_internal_cv(make_net, train_p_obj.patientList, temporal_feats,
                        encoder, stats, K=args.cv, lr=args.lrs[0],
                        epochs=args.epochs, wd=args.wd)


def run_internal_cv(make_net, train_patients, temporal_feats, encoder, stats,
                    K, lr, epochs, wd, eval_every=5):
    """K-fold CV inside train. For each held-out inner fold, train on the rest
    and collect out-of-fold val predictions. Aggregate AUPR over ALL positives.

    This does NOT touch the outer val/test. It is purely a lower-noise estimate
    of pretrain quality, to separate metric noise from real instability.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    pats = list(train_patients)
    labels = np.array([
        1 if getattr(p, "akdPositive", False) else 0 for p in pats])
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)

    # We snapshot OOF predictions at a few epoch checkpoints to see a curve.
    checkpoints = list(range(eval_every, epochs + 1, eval_every))
    oof = {ep: np.full(len(pats), np.nan) for ep in checkpoints}

    for ki, (tr_idx, va_idx) in enumerate(skf.split(pats, labels)):
        tr_p = [pats[i] for i in tr_idx]
        va_p = [pats[i] for i in va_idx]
        tr_ds = HybridDataset(tr_p, temporal_feats, encoder, stats)
        va_ds = HybridDataset(va_p, temporal_feats, encoder, stats)
        tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=hybrid_collate_fn)
        va_ld = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)

        torch.manual_seed(0); np.random.seed(0)
        net = make_net().to(DEVICE)
        head = SupervisedHead(net.latent_dim + len(FIXED_FEATURES)).to(DEVICE)
        opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()),
                               lr=lr, weight_decay=wd)
        crit = nn.BCELoss()
        for epoch in range(epochs):
            net.train(); head.train()
            for t_data, lbl, s_data in tr_ld:
                lbl = lbl.to(DEVICE); s_data = s_data.to(DEVICE)
                z, _, _ = net(t_data, deterministic=True)
                preds = head(torch.cat([z, s_data], dim=1)).squeeze(-1)
                loss = crit(preds, lbl)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(net.parameters()) + list(head.parameters()), 1.0)
                opt.step()
            if (epoch + 1) in oof:
                net.eval(); head.eval()
                preds_all = []
                with torch.no_grad():
                    for t_data, lbl, s_data in va_ld:
                        s_data = s_data.to(DEVICE)
                        z, _, _ = net(t_data, deterministic=True)
                        p = head(torch.cat([z, s_data], dim=1)).squeeze(-1)
                        preds_all.extend(p.cpu().numpy())
                oof[(epoch + 1)][va_idx] = np.array(preds_all)
        print(f"  inner fold {ki+1}/{K} done")

    print(f"\n{'ep':>3} | {'OOF_AUPR':>8} | {'OOF_AUC':>7} | (pos={int(labels.sum())}/{len(labels)})")
    for ep in checkpoints:
        p = oof[ep]
        m = ~np.isnan(p)
        aupr = average_precision_score(labels[m], p[m])
        try: auc = roc_auc_score(labels[m], p[m])
        except ValueError: auc = float("nan")
        print(f"{ep:3d} | {aupr:8.4f} | {auc:7.4f} |")
    print("\n  Stable OOF curve here + wobbly small-val curve above => cause B.")


if __name__ == "__main__":
    main()