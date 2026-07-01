OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nohup python -u debug_rl_rloo.py \
  --holdout --reps 10 --K 4 --rl_epochs 80 --eval_every 4 \
  --clf tabpfn --lr 1e-3 --stop_metric AUC \
  > rloo_k10.log 2>&1 &
echo "PID $!"


python debug_rl_onefold.py --reps 10 --K 4 --rl_epochs 40 --eval_every 4     --clf xgb --lr 1e-2 --stop_metric none --no_normalize --fold_id 0
python debug_rl_rloo.py --holdout  --reps 10 --K 4 --rl_epochs 40   --eval_every 4 --clf xgb --lr 1e-2 --stop_metric none --no_normalize      

================  SUMMARY (RLOO scratch, 10-fold on remainder, early-stop on val-none, scored on FROZEN hold-out)  ================
    unit   ep |  S+L AUC   +Z AUC   dZ_AU |  S+L AUPR  +Z AUPR   dZ_AP | final_dAU
   fold0   40 |   0.7935   0.8284 +0.0349 |    0.4832   0.5733 +0.0901 |   +0.0349
   fold1   40 |   0.8213   0.8403 +0.0191 |    0.5863   0.6072 +0.0209 |   +0.0191
   fold2   40 |   0.8129   0.8132 +0.0003 |    0.5573   0.5365 -0.0208 |   +0.0003
   fold3   40 |   0.8163   0.8201 +0.0037 |    0.5392   0.5819 +0.0428 |   +0.0037
   fold4   40 |   0.8132   0.8269 +0.0137 |    0.5211   0.5912 +0.0702 |   +0.0137
   fold5   40 |   0.8117   0.8305 +0.0188 |    0.5405   0.5497 +0.0092 |   +0.0188
   fold6   40 |   0.8198   0.8145 -0.0052 |    0.5000   0.5517 +0.0518 |   -0.0052
   fold7   40 |   0.8226   0.8256 +0.0030 |    0.5617   0.5712 +0.0096 |   +0.0030
   fold8   40 |   0.8242   0.8238 -0.0004 |    0.5662   0.5707 +0.0045 |   -0.0004
   fold9   40 |   0.8156   0.8219 +0.0063 |    0.5387   0.5756 +0.0368 |   +0.0063

  At val-selected epoch, across 10 folds:
    AUC :  S+L 0.8151±0.0083 | S+L+Z 0.8245±0.0075 | dZ +0.0094 wins 8/10
    AUPR:  S+L 0.5394±0.0296 | S+L+Z 0.5709±0.0197 | dZ +0.0315 wins 9/10

  early-stop vs run-to-end (dZ-AUC): selected +0.0094 | final-epoch +0.0094 | gain +0.0000

  PAIRED TEST  (S+L+Z) vs (S+L)  at val-selected epoch, 10 folds:
    AUC-ROC        mean dZ +0.0094 | wins 8/10 | paired-t t=+2.450 p=0.0367 | Wilcoxon W=7.0 p=0.0371 | dz=+0.77
    AUPR           mean dZ +0.0315 | wins 9/10 | paired-t t=+2.982 p=0.0154 | Wilcoxon W=4.0 p=0.0137 | dz=+0.94
    (dz = Cohen's d for paired diffs; |dz|>0.8 large. n folds is small,
     so Wilcoxon p floors at ~0.002 for n=10 / ~0.06 for n=5.)

[full-fit base] fit on ALL remainder (N=798, no CV, val included) -> score on FROZEN hold-out (N=199)

================  FULL-FIT BASE (all remainder, no CV)  ================
  fit N=798 (pos 174, rate 0.218) | test N=199 (pos 43, rate 0.216)
  S+L    :  AUC 0.8229  AUPR 0.5596
  S+L+Z  :  AUC 0.8095  AUPR 0.5146   (dZ AUC -0.0134  dZ AUPR -0.0450)