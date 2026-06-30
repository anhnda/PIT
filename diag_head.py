"""
Why does an encoder pretrained through a WEAK MLP head transfer well to strong
tabular classifiers, while RL (direct optimization for the classifier) hurts?

Hypothesis: the surrogate head acts as a regularizer on the representation.
A simple/weak head forces z to be separable by a SMOOTH function, which is a
universal structure every strong tabular classifier exploits well. A strong
head lets z overfit to that head's idiosyncrasies, transferring worse. RL is
the extreme of overfitting to one (noisy, shifting) classifier boundary.

Test: pretrain the SAME pooled encoder through heads of increasing strength,
then measure transfer = dZ over (S+L) via OOF on real classifiers.

  head=linear : logistic (z+static -> 1 linear layer)         [simplest]
  head=weak   : the paper's SupervisedHead (128->64->1, drop)  [current]
  head=strong : deeper/wider, low dropout                      [strongest]

Prediction if hypothesis holds: dZ(linear) >= dZ(weak) > dZ(strong).
If instead dZ(strong) is best, the hypothesis is wrong.

Run:  python diag_head.py --clf xgb tabpfn --heads linear weak strong
"""
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from TabPFNRL import (
    FIXED_FEATURES, HybridDataset, hybrid_collate_fn, SimpleStaticEncoder,
)
from pooled_encoder import PooledRNNPolicyNetwork
from TimeEmbedding import DEVICE
from TimeEmbeddingVal import (
    get_all_temporal_features, split_patients_train_val, load_and_prepare_patients,
)
from utils.prepare_data import trainTestPatients


# ---- heads of varying strength -------------------------------------------
class LinearHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__(); self.fc = nn.Linear(in_dim, 1)
    def forward(self, x): return torch.sigmoid(self.fc(x)).squeeze(-1)


class WeakHead(nn.Module):  # == paper's SupervisedHead
    def __init__(self, in_dim, h=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h); self.bn1 = nn.BatchNorm1d(h)
        self.fc2 = nn.Linear(h, h // 2); self.bn2 = nn.BatchNorm1d(h // 2)
        self.fc3 = nn.Linear(h // 2, 1); self.drop = nn.Dropout(0.3)
    def forward(self, x):
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        return torch.sigmoid(self.fc3(x)).squeeze(-1)


class StrongHead(nn.Module):  # deeper, wider, low dropout -> can overfit z
    def __init__(self, in_dim, h=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(h // 2, 1),
        )
    def forward(self, x): return torch.sigmoid(self.net(x)).squeeze(-1)


class GatedDecisionHead(nn.Module):
    """Gated head meant to mimic tree-classifier (XGBoost/CatBoost) logic:
    a sigmoid feature gate (soft feature selection like a tree's splits) +
    GLU layers (conditional gating) + residual. Tests whether pretraining the
    encoder through a head shaped like the downstream tree classifier yields a
    z that transfers better to XGBoost/CatBoost than a plain MLP head."""
    def __init__(self, in_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(in_dim, in_dim), nn.Sigmoid())
        self.fc1 = nn.Linear(in_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.final = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = x * self.gate(x)
        out = F.glu(self.fc1(x), dim=-1)
        out = self.dropout(out)
        residual = out
        out = F.glu(self.fc2(out), dim=-1)
        out = out + residual
        return torch.sigmoid(self.final(out)).squeeze(-1)


class PlainHead(nn.Module):
    """Standard MLP: Linear-ReLU-Dropout-Linear-ReLU-Linear. No BatchNorm,
    no GLU, no gating. Clean control to isolate what makes the 'weak' head
    transfer well — its BatchNorm, or just being a mid-size MLP."""
    def __init__(self, in_dim, h=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h // 2, 1),
        )
    def forward(self, x): return torch.sigmoid(self.net(x)).squeeze(-1)


class PlainBNHead(nn.Module):
    """PlainHead + BatchNorm. Identical to PlainHead in size/depth/dropout;
    the ONLY difference is BatchNorm after each linear. If plain_bn ~ weak,
    BatchNorm (not architecture/capacity) is what makes z transfer well."""
    def __init__(self, in_dim, h=128, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h);     self.bn1 = nn.BatchNorm1d(h)
        self.fc2 = nn.Linear(h, h // 2);     self.bn2 = nn.BatchNorm1d(h // 2)
        self.fc3 = nn.Linear(h // 2, 1);     self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        return torch.sigmoid(self.fc3(x)).squeeze(-1)


class LinearBNHead(nn.Module):
    """Logistic head but with a BatchNorm on the input features first. Tests
    whether BN alone — even with the simplest possible classifier — is enough
    to make z transfer well."""
    def __init__(self, in_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.fc = nn.Linear(in_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.fc(self.bn(x))).squeeze(-1)


HEADS = {"linear": LinearHead, "weak": WeakHead, "strong": StrongHead,
         "gated": GatedDecisionHead, "plain": PlainHead,
         "plain_bn": PlainBNHead, "linear_bn": LinearBNHead}


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


def pretrain(net, head_name, loader, epochs, lr=1e-3):
    in_dim = net.latent_dim + len(FIXED_FEATURES)
    head = HEADS[head_name](in_dim).to(DEVICE)
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=lr)
    crit = nn.BCELoss()
    for _ in range(epochs):
        net.train(); head.train()
        for t, lbl, s in loader:
            lbl = lbl.to(DEVICE); s = s.to(DEVICE)
            z, _, _ = net(t, deterministic=True)
            p = head(torch.cat([z, s], dim=1))
            loss = crit(p, lbl)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(net.parameters()) + list(head.parameters()), 1.0)
            opt.step()
    net.eval(); return net


def blocks(net, patients, feats, enc, stats):
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


def oof(b, Y, spec, clf, K=5, seed=0):
    X = np.hstack([b[k] for k in spec])
    ratio = float((Y == 0).sum()) / max(int((Y == 1).sum()), 1)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    p = np.full(len(Y), np.nan)
    for tr, va in skf.split(X, Y):
        c = make_clf(clf, ratio); c.fit(X[tr], Y[tr]); p[va] = c.predict_proba(X[va])[:, 1]
    return average_precision_score(Y, p), roc_auc_score(Y, p)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    pa.add_argument("--clf", nargs="+", default=["xgb", "cat"])
    pa.add_argument("--heads", nargs="+",
                    default=["weak", "plain", "plain_bn", "linear", "linear_bn"],
                    choices=["linear", "weak", "strong", "gated", "plain",
                             "plain_bn", "linear_bn"])
    pa.add_argument("--epochs", type=int, default=20)
    pa.add_argument("--seed", type=int, default=27)
    args = pa.parse_args()

    patients = load_and_prepare_patients()
    feats = get_all_temporal_features(patients)
    enc = SimpleStaticEncoder(FIXED_FEATURES); enc.fit(patients.patientList)
    all_folds = list(trainTestPatients(patients, seed=args.seed))

    # dZ[clf][head] = list of (dAUPR, dAUC); zalone[clf][head] = list of AUPR
    dZ = {c: {h: [] for h in args.heads} for c in args.clf}
    zalone = {c: {h: [] for h in args.heads} for c in args.clf}

    for fi in args.folds:
        train_full, _ = all_folds[fi]
        tr_obj, _ = split_patients_train_val(train_full, val_ratio=0.1, seed=42 + fi)
        tp = tr_obj.patientList
        stats = HybridDataset(tp, feats, enc).get_normalization_stats()
        loader = DataLoader(HybridDataset(tp, feats, enc, stats), batch_size=32,
                            shuffle=True, collate_fn=hybrid_collate_fn)
        for h in args.heads:
            torch.manual_seed(0); np.random.seed(0)
            net = make_net(len(feats)).to(DEVICE)
            net = pretrain(net, h, loader, args.epochs)
            b, Y = blocks(net, tp, feats, enc, stats)
            for c in args.clf:
                a0, c0 = oof(b, Y, ["S", "L"], c, seed=args.seed)
                a1, c1 = oof(b, Y, ["S", "L", "Z"], c, seed=args.seed)
                dZ[c][h].append((a1 - a0, c1 - c0))
                az, _ = oof(b, Y, ["Z"], c, seed=args.seed)  # z alone -> leak check
                zalone[c][h].append(az)
        print(f"fold {fi} done", flush=True)

    def ms(vals, i):
        a = np.array([v[i] for v in vals]); return a.mean(), a.std()

    for c in args.clf:
        print(f"\n=== transfer dZ = (S+L+Z)-(S+L), clf={c}, {len(args.folds)} folds ===")
        print(f"{'head':<8} | {'dAUPR mean±std':>18} | {'dAUC mean±std':>18}")
        print("-" * 50)
        print(f"{'head':<10} | {'dAUPR mean±std':>18} | {'Z-alone AUPR':>13} | flag")
        print("-" * 60)
        for h in args.heads:
            am, asd = ms(dZ[c][h], 0); cm, csd = ms(dZ[c][h], 1)
            za = np.mean(zalone[c][h])
            fa = "OK" if abs(am) > asd else "noise"
            # z alone should be modest (z complements static). If z alone is very
            # high, z has absorbed the label directly -> leak, not learning.
            leak = "  <-- LEAK? z alone too predictive" if za > 0.75 else ""
            print(f"{h:<10} | {am:+.4f} ± {asd:.4f} [{fa:>5}] | {za:13.4f} | {leak}")
        print("  Z-alone is z predicting the label by itself. Modest = z complements")
        print("  static. Very high (>0.75) = label leaked into z during pretrain.")


if __name__ == "__main__":
    main()