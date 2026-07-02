"""
debug_rl_wu.py  --  debug_rl_rloo.py + LR warm-up/decay schedule.

Inherits EVERYTHING from debug_rl_rloo (all CLI options, run_holdout, run_folds,
report, full-fit base, cross-seed, --log, ...). The only change is the policy
optimizer's learning rate follows a one-cycle-style schedule:

    phase 1 (first --warm_up fraction of epochs):  1e-6  ->  --lr   (warm-up)
    phase 2 (remaining epochs):                    --lr  ->  1e-6   (decay)

--lr is the PEAK lr (top of the warm-up ramp). --warm_up defaults to 0.5, i.e.
ramp up for the first half of training, decay for the second half. Both ramps
are cosine. Set --warm_up 0 for pure decay from peak; --warm_up 1 for pure ramp.

Usage (same command line as debug_rl_rloo.py, plus --warm_up):
    python debug_rl_wu.py --holdout --reps 10 --K 4 --rl_epochs 40 \
        --eval_every 4 --clf cat --lr 1e-2 --stop_metric none --no_normalize \
        --warm_up 0.5
"""
import argparse, sys, numpy as np, torch
from torch.utils.data import DataLoader

import debug_rl_rloo as D
from debug_rl_rloo import (
    DEVICE, blocks_for_eval, test_dZ, oof_reward,
    HybridDataset, hybrid_collate_fn,
)

LR_FLOOR = 1e-6          # bottom of both ramps
WARMUP_FRAC = 0.5        # set by main() from --warm_up before training


def _lr_multiplier(step, total_steps, warm_steps, peak):
    """Return lr as an absolute value for a given global step.
    Cosine ramp LR_FLOOR->peak over warm_steps, then cosine peak->LR_FLOOR."""
    if total_steps <= 1:
        return peak
    if warm_steps > 0 and step < warm_steps:
        frac = step / max(warm_steps, 1)                      # 0 -> 1
        return LR_FLOOR + (peak - LR_FLOOR) * 0.5 * (1 - np.cos(np.pi * frac))
    # decay phase
    dsteps = max(total_steps - warm_steps, 1)
    frac = (step - warm_steps) / dsteps                      # 0 -> 1
    return LR_FLOOR + (peak - LR_FLOOR) * 0.5 * (1 + np.cos(np.pi * frac))


def rl_rloo_wu(net, tr_p, te_p, feats, enc, stats, clf, epochs, eval_every,
               K=4, lr=1e-3, ent_coef=0.01, whiten=True, val_p=None):
    """Verbatim copy of debug_rl_rloo.rl_rloo with a warm-up/decay LR schedule
    applied per optimizer step. `lr` is the PEAK learning rate."""
    ref_b, _ = blocks_for_eval(net, tr_p, feats, enc, stats)
    z_ref = ref_b["Z"].copy()

    ds = HybridDataset(tr_p, feats, enc, stats)
    ld = DataLoader(ds, batch_size=len(tr_p), shuffle=False,
                    collate_fn=hybrid_collate_fn)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # ---- schedule setup: one optimizer step per epoch (batch = full train) ----
    peak = lr
    total_steps = epochs
    warm_steps = int(round(WARMUP_FRAC * epochs))
    print(f"    [wu] warm_up={WARMUP_FRAC:g} peak_lr={peak:g} floor={LR_FLOOR:g} "
          f"| ramp {warm_steps}ep -> decay {epochs-warm_steps}ep", flush=True)

    run_mean, run_var, run_n = 0.0, 1.0, 0
    log = []

    def snapshot(ep, rstats):
        net.eval()
        m = test_dZ(net, tr_p, te_p, feats, enc, stats, clf)
        drift = float(np.linalg.norm(m["trainZ"] - z_ref) / np.sqrt(len(z_ref)))
        if val_p is not None:
            mv = test_dZ(net, tr_p, val_p, feats, enc, stats, clf)
            val_dap, val_dau = mv["dAUPR"], mv["dAUC"]
        else:
            val_dap, val_dau = np.nan, np.nan
        log.append(dict(epoch=ep, test_dAUPR=m["dAUPR"], test_dAUC=m["dAUC"],
                        base_AUPR=m["base_AUPR"], base_AUC=m["base_AUC"],
                        full_AUPR=m["full_AUPR"], full_AUC=m["full_AUC"],
                        val_dAUPR=val_dap, val_dAUC=val_dau,
                        drift=drift, **rstats))

    snapshot(0, dict(r_mean=np.nan, r_pos=np.nan, r_neg=np.nan,
                     grad=np.nan, A_pos=np.nan, A_neg=np.nan))

    global_step = 0
    for ep in range(1, epochs + 1):
        net.train()
        rstats = {}
        for t, lbl, s in ld:
            # ---- set LR for this step from the warm-up/decay schedule ----
            cur_lr = _lr_multiplier(global_step, total_steps, warm_steps, peak)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            global_step += 1

            s = s.to(DEVICE)
            Y = lbl.numpy().astype(int)
            N = len(Y)

            vals = t['values'].cpu().numpy(); masks = t['masks'].cpu().numpy()
            L = np.zeros((N, vals.shape[2]), dtype=np.float32)
            for i in range(N):
                for f in range(vals.shape[2]):
                    idx = np.where(masks[i, :, f] > 0)[0]
                    if len(idx):
                        L[i, f] = vals[i, idx[-1], f]
            s_np = s.detach().cpu().numpy()

            logps = []
            R = np.empty((K, N), dtype=float)
            z_last = None
            for k in range(K):
                z, logp, mean = net(t, deterministic=False, temperature=1.0)
                logps.append(logp)
                z_last = z
                z_np = z.detach().cpu().numpy()
                Xz = np.hstack([s_np, L, z_np])
                R[k] = oof_reward(Xz, Y, clf, seed=k)

            if whiten:
                batch_mean = R.mean(); batch_var = R.var()
                run_n += 1
                run_mean += (batch_mean - run_mean) / run_n
                run_var += (batch_var - run_var) / run_n
                A = (R - run_mean) / (np.sqrt(run_var) + 1e-8)
            else:
                A = (R - R.mean(axis=1, keepdims=True)) / \
                    (R.std(axis=1, keepdims=True) + 1e-8)

            A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)
            loss_terms = [-(logps[k] * A_t[k]).mean() for k in range(K)]
            policy_loss = torch.stack(loss_terms).mean()

            ent = 0.5 * torch.log(2 * np.pi * np.e * (z_last.var(dim=0) + 1e-6)).sum()
            loss = policy_loss - ent_coef * ent

            opt.zero_grad(); loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            Rm = R.mean(0)
            rstats = dict(r_mean=float(R.mean()),
                          r_pos=float(Rm[Y == 1].mean()) if (Y == 1).any() else np.nan,
                          r_neg=float(Rm[Y == 0].mean()) if (Y == 0).any() else np.nan,
                          A_pos=float(A[:, Y == 1].mean()) if (Y == 1).any() else np.nan,
                          A_neg=float(A[:, Y == 0].mean()) if (Y == 0).any() else np.nan)
        if ep % eval_every == 0 or ep == epochs:
            rstats["grad"] = float(gnorm)
            rstats["lr"] = float(cur_lr)
            snapshot(ep, rstats)
    net.eval()
    return log


def main():
    # parse only --warm_up here, leave the rest for D.main()'s own parser
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--warm_up", type=float, default=0.5,
                     help="fraction of epochs spent ramping lr 1e-6 -> --lr "
                          "before decaying back (default 0.5)")
    known, rest = pre.parse_known_args()

    global WARMUP_FRAC
    WARMUP_FRAC = known.warm_up
    if not (0.0 <= WARMUP_FRAC <= 1.0):
        raise SystemExit("--warm_up must be in [0, 1]")

    # inject the schedule into the inherited pipeline
    D.rl_rloo = rl_rloo_wu

    # hand the remaining args to the inherited main (so ALL options are inherited)
    sys.argv = [sys.argv[0]] + rest
    D.main()


if __name__ == "__main__":
    main()