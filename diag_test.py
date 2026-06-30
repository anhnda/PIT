"""
Honest evaluation: train on the fold's TRAIN, evaluate on the fold's held-out
TEST. The encoder is pretrained ONLY on train (checkpoint chosen on a val split
carved from train); z for the test patients comes from an encoder that never
saw them. Then a classifier is fit on train-[features] and scored on test.

This replaces the earlier diagnostics, which (wrongly) pretrained on all of
train and then did OOF *inside train* — that let the label leak through the
encoder and never touched the test set. Numbers from those are not valid.

Reports, per config, dZ = (S+L+Z) - (S+L) and Z-alone, on TEST, mean±std over
the 5 outer folds. Test folds are small (~200 patients, ~40-50 positives), so
expect larger std than the (invalid) in-train OOF numbers.

Configs: encoder in {final, pool}; head in {weak, plain, plain_bn, linear,
linear_bn, gated, strong}.

Run:  python diag_test.py --clf xgb --encoder pool --head weak
      python diag_test.py --clf xgb cat --encoder pool --head weak linear_bn
"""
import argparse, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from TabPFNRL import (
    FIXED_FEATURES, RNNPolicyNetwork, HybridDataset, hybrid_collate_fn,
    SimpleStaticEncoder,
)
from pooled_encoder import PooledRNNPolicyNetwork
from TimeEmbedding import DEVICE
from TimeEmbeddingVal import (
    get_all_temporal_features, split_patients_train_val, load_and_prepare_patients,
)
from utils.prepare_data import trainTestPatients


# ---------- heads ----------
class LinearHead(nn.Module):
    def __init__(s, d): super().__init__(); s.fc = nn.Linear(d, 1)
    def forward(s, x): return torch.sigmoid(s.fc(x)).squeeze(-1)

class WeakHead(nn.Module):
    def __init__(s, d, h=128):
        super().__init__()
        s.fc1=nn.Linear(d,h); s.bn1=nn.BatchNorm1d(h)
        s.fc2=nn.Linear(h,h//2); s.bn2=nn.BatchNorm1d(h//2)
        s.fc3=nn.Linear(h//2,1); s.drop=nn.Dropout(0.3)
    def forward(s,x):
        x=s.drop(F.relu(s.bn1(s.fc1(x)))); x=s.drop(F.relu(s.bn2(s.fc2(x))))
        return torch.sigmoid(s.fc3(x)).squeeze(-1)

class PlainHead(nn.Module):
    def __init__(s, d, h=128, p=0.3):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(p),
                            nn.Linear(h,h//2),nn.ReLU(),nn.Dropout(p),nn.Linear(h//2,1))
    def forward(s,x): return torch.sigmoid(s.net(x)).squeeze(-1)

class PlainBNHead(nn.Module):
    def __init__(s, d, h=128, p=0.3):
        super().__init__()
        s.fc1=nn.Linear(d,h); s.bn1=nn.BatchNorm1d(h)
        s.fc2=nn.Linear(h,h//2); s.bn2=nn.BatchNorm1d(h//2)
        s.fc3=nn.Linear(h//2,1); s.drop=nn.Dropout(p)
    def forward(s,x):
        x=s.drop(F.relu(s.bn1(s.fc1(x)))); x=s.drop(F.relu(s.bn2(s.fc2(x))))
        return torch.sigmoid(s.fc3(x)).squeeze(-1)

class LinearBNHead(nn.Module):
    def __init__(s, d):
        super().__init__(); s.bn=nn.BatchNorm1d(d); s.fc=nn.Linear(d,1)
    def forward(s,x): return torch.sigmoid(s.fc(s.bn(x))).squeeze(-1)

HEADS = {"linear":LinearHead,"weak":WeakHead,"plain":PlainHead,
         "plain_bn":PlainBNHead,"linear_bn":LinearBNHead}


def make_clf(name, ratio):
    if name=="xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,
            subsample=0.8,colsample_bytree=0.8,scale_pos_weight=ratio,
            random_state=42,eval_metric="auc")
    if name=="cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=200,depth=4,learning_rate=0.05,
            loss_function="Logloss",eval_metric="AUC",scale_pos_weight=ratio,
            random_seed=42,verbose=False,allow_writing_files=False,task_type="CPU")
    if name=="tabpfn":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
    raise ValueError(name)


def make_net(enc_kind, n_feat):
    Net = PooledRNNPolicyNetwork if enc_kind=="pool" else RNNPolicyNetwork
    return Net(input_dim=n_feat, hidden_dim=20, latent_dim=28, time_dim=32)


def pretrain(net, head_name, loader, val_loader, epochs, lr=1e-3):
    """Pretrain on TRAIN only; keep the checkpoint with best val AUPR
    (val carved from train). Encoder never sees test."""
    in_dim = net.latent_dim + len(FIXED_FEATURES)
    head = HEADS[head_name](in_dim).to(DEVICE)
    opt = torch.optim.Adam(list(net.parameters())+list(head.parameters()), lr=lr)
    crit = nn.BCELoss()
    best_aupr, best_state = -1, None
    for ep in range(epochs):
        net.train(); head.train()
        for t, lbl, s in loader:
            lbl=lbl.to(DEVICE); s=s.to(DEVICE)
            z,_,_ = net(t, deterministic=True)
            p = head(torch.cat([z,s],dim=1))
            loss = crit(p, lbl)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1.0)
            opt.step()
        # val checkpoint
        net.eval(); head.eval()
        ps, ys = [], []
        with torch.no_grad():
            for t, lbl, s in val_loader:
                s=s.to(DEVICE); z,_,_ = net(t, deterministic=True)
                p = head(torch.cat([z,s],dim=1))
                ps.extend(p.cpu().numpy()); ys.extend(lbl.numpy())
        au = average_precision_score(ys, ps) if len(set(ys))>1 else 0
        if au > best_aupr:
            best_aupr = au; best_state = copy.deepcopy(net.state_dict())
    if best_state is not None: net.load_state_dict(best_state)
    net.eval(); return net


def blocks(net, patients, feats, enc, stats):
    ds = HybridDataset(patients, feats, enc, stats)
    ld = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
    S,L,MS,Z,Y = [],[],[],[],[]
    with torch.no_grad():
        for t, lbl, s in ld:
            _,_,mean = net(t, deterministic=True); Z.append(mean.cpu().numpy())
            vals=t['values'].cpu().numpy(); masks=t['masks'].cpu().numpy()
            for i in range(len(vals)):
                last=[]; mn=[]; sd=[]
                for f in range(vals.shape[2]):
                    idx=np.where(masks[i,:,f]>0)[0]
                    if len(idx):
                        vv=vals[i,idx,f]
                        last.append(vv[-1]); mn.append(float(np.mean(vv)))
                        sd.append(float(np.std(vv)) if len(vv)>1 else 0.0)
                    else:
                        last.append(0.0); mn.append(0.0); sd.append(0.0)
                L.append(last); MS.append(mn+sd)
            S.append(s.numpy()); Y.extend(lbl.numpy())
    return {"S":np.vstack(S),"L":np.array(L),"MS":np.array(MS),
            "Z":np.vstack(Z)}, np.array(Y)


def fit_score(tr_b, tr_Y, te_b, te_Y, spec, clf):
    Xtr=np.hstack([tr_b[k] for k in spec]); Xte=np.hstack([te_b[k] for k in spec])
    ratio=float((tr_Y==0).sum())/max(int((tr_Y==1).sum()),1)
    c=make_clf(clf, ratio); c.fit(Xtr, tr_Y)
    p=c.predict_proba(Xte)[:,1]
    return average_precision_score(te_Y,p), roc_auc_score(te_Y,p)


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--folds", type=int, nargs="+", default=[0,1,2,3,4])
    pa.add_argument("--clf", nargs="+", default=["xgb"])
    pa.add_argument("--encoder", default="pool", choices=["final","pool"])
    pa.add_argument("--head", nargs="+", default=["weak"],
                    choices=list(HEADS.keys()))
    pa.add_argument("--epochs", type=int, default=20)
    pa.add_argument("--seed", type=int, default=27)
    args=pa.parse_args()

    patients=load_and_prepare_patients()
    feats=get_all_temporal_features(patients)
    enc=SimpleStaticEncoder(FIXED_FEATURES); enc.fit(patients.patientList)
    folds=list(trainTestPatients(patients, seed=args.seed))

    # res[clf][head] = list per fold of dict(dAUPR,dAUC,zaupr, base_aupr, full_aupr)
    res={c:{h:[] for h in args.head} for c in args.clf}

    for fi in args.folds:
        train_full, test_p = folds[fi]
        tr_obj, val_obj = split_patients_train_val(train_full, val_ratio=0.1, seed=42+fi)
        stats = HybridDataset(tr_obj.patientList, feats, enc).get_normalization_stats()
        tr_loader = DataLoader(HybridDataset(tr_obj.patientList,feats,enc,stats),
                               batch_size=32, shuffle=True, collate_fn=hybrid_collate_fn)
        val_loader = DataLoader(HybridDataset(val_obj.patientList,feats,enc,stats),
                                batch_size=32, shuffle=False, collate_fn=hybrid_collate_fn)
        for h in args.head:
            torch.manual_seed(0); np.random.seed(0)
            net = make_net(args.encoder, len(feats)).to(DEVICE)
            net = pretrain(net, h, tr_loader, val_loader, args.epochs)
            tr_b, tr_Y = blocks(net, train_full.patientList, feats, enc, stats)
            te_b, te_Y = blocks(net, test_p.patientList, feats, enc, stats)
            for c in args.clf:
                row = {}
                for name, spec in [("S",["S"]),("S+L",["S","L"]),
                                   ("S+L+MS",["S","L","MS"]),
                                   ("S+L+Z",["S","L","Z"]),
                                   ("S+L+MS+Z",["S","L","MS","Z"])]:
                    a,r = fit_score(tr_b,tr_Y,te_b,te_Y,spec,c)
                    row[name] = a          # AUPR
                    row[name+"@auc"] = r   # AUC-ROC
                za,zr = fit_score(tr_b,tr_Y,te_b,te_Y,["Z"],c)
                row["Zalone"] = za
                row["Zalone@auc"] = zr
                res[c][h].append(row)
        print(f"fold {fi}: test n={len(te_Y)} pos={int(te_Y.sum())}", flush=True)

    for c in args.clf:
        print(f"\n=== TEST (AUC-ROC | AUPR), clf={c}, encoder={args.encoder}, {len(args.folds)} folds ===")
        for h in args.head:
            rows = res[c][h]
            print(f"\n  head={h}")
            cfgs = ["S","S+L","S+L+MS","S+L+Z","S+L+MS+Z"]
            print(f"    {'':<10} {'AUC-ROC':<18} {'AUPR':<18}")
            for name in cfgs:
                au = np.array([r[name+"@auc"] for r in rows])
                ap = np.array([r[name] for r in rows])
                print(f"    {name:<10} {au.mean():.4f} ± {au.std():.4f}    "
                      f"{ap.mean():.4f} ± {ap.std():.4f}")
            # key deltas on test, paired per-fold, for a given metric suffix
            def dl(a, b, suf=""):
                arr_a = np.array([r[a+suf] for r in rows])
                arr_b = np.array([r[b+suf] for r in rows])
                d = arr_a - arr_b
                wins = int((d > 0).sum()); n = len(d)
                pstd = d.std(ddof=1) if n > 1 else 0.0
                tstat = d.mean() / (pstd/np.sqrt(n) + 1e-9)
                sig = "[OK]" if abs(tstat) >= 2.0 else "[weak]"
                return (f"{d.mean():+.4f} | per-fold: {[f'{x:+.4f}' for x in d]} "
                        f"| wins {wins}/{n} | t={tstat:+.2f} {sig}")
            for metric, suf in [("AUC-ROC", "@auc"), ("AUPR", "")]:
                za = np.array([r["Zalone"+suf] for r in rows]).mean()
                print(f"    --- deltas [{metric}] (test, PAIRED per-fold) ---")
                print(f"    MS over S+L      (S+L+MS)-(S+L)   : {dl('S+L+MS','S+L',suf)}")
                print(f"    Z  over S+L      (S+L+Z)-(S+L)    : {dl('S+L+Z','S+L',suf)}")
                print(f"    Z over S+L+MS  (S+L+MS+Z)-(S+L+MS): {dl('S+L+MS+Z','S+L+MS',suf)}")
                print(f"    Z-alone {metric}: {za:.4f}")


if __name__=="__main__":
    main()