python debug_rl_rloo.py --holdout --reps 10 --K 4 --rl_epochs 40 --eval_every 4 \
    --clf cat --lr 1e-2 --stop_metric none --no_normalize --neutral_init


python debug_rl_rloo.py --holdout --reps 10 --K 4 --rl_epochs 40 --eval_every 4 \
    --clf xgb --lr 1e-2 --stop_metric none --no_normalize --neutral_init

================  SUMMARY (RLOO scratch, 10-fold on remainder, early-stop on val-none, scored on FROZEN hold-out)  ================
    unit   ep |  S+L AUC   +Z AUC   dZ_AU |  S+L AUPR  +Z AUPR   dZ_AP | final_dAU
   fold0   40 |   0.7935   0.8177 +0.0242 |    0.4832   0.5203 +0.0371 |   +0.0242
   fold1   40 |   0.8213   0.8383 +0.0170 |    0.5863   0.5960 +0.0097 |   +0.0170
   fold2   40 |   0.8129   0.8189 +0.0060 |    0.5573   0.5610 +0.0037 |   +0.0060
   fold3   40 |   0.8163   0.8241 +0.0078 |    0.5392   0.6031 +0.0640 |   +0.0078
   fold4   40 |   0.8132   0.8436 +0.0304 |    0.5211   0.6111 +0.0900 |   +0.0304
   fold5   40 |   0.8117   0.8274 +0.0157 |    0.5405   0.5913 +0.0509 |   +0.0157
   fold6   40 |   0.8198   0.8478 +0.0280 |    0.5000   0.5671 +0.0671 |   +0.0280
   fold7   40 |   0.8226   0.8363 +0.0137 |    0.5617   0.6111 +0.0494 |   +0.0137
   fold8   40 |   0.8242   0.8347 +0.0104 |    0.5662   0.5923 +0.0260 |   +0.0104
   fold9   40 |   0.8156   0.8175 +0.0019 |    0.5387   0.5118 -0.0270 |   +0.0019

  At val-selected epoch, across 10 folds:
    AUC :  S+L 0.8151±0.0083 | S+L+Z 0.8306±0.0105 | dZ +0.0155 wins 10/10
    AUPR:  S+L 0.5394±0.0296 | S+L+Z 0.5765±0.0341 | dZ +0.0371 wins 9/10


python debug_rl_rloo.py --holdout --reps 10 --K 4 --rl_epochs 10 --eval_every 2     --clf tabpfn --lr 1e-4 --stop_metric none --no_normalize --neutral_init

================  SUMMARY (RLOO scratch, 10-fold on remainder, early-stop on val-none, scored on FROZEN hold-out)  ================
    unit   ep |  S+L AUC   +Z AUC   dZ_AU |  S+L AUPR  +Z AUPR   dZ_AP | final_dAU
   fold0   10 |   0.8102   0.8205 +0.0103 |    0.4987   0.5138 +0.0151 |   +0.0103
   fold1   10 |   0.8134   0.8233 +0.0099 |    0.5370   0.5354 -0.0016 |   +0.0099
   fold2   10 |   0.8070   0.8204 +0.0133 |    0.5148   0.5255 +0.0106 |   +0.0133
   fold3   10 |   0.8045   0.8123 +0.0078 |    0.5082   0.5004 -0.0079 |   +0.0078
   fold4   10 |   0.8151   0.8211 +0.0060 |    0.5206   0.5186 -0.0020 |   +0.0060
   fold5   10 |   0.8049   0.8134 +0.0085 |    0.5287   0.5412 +0.0126 |   +0.0085
   fold6   10 |   0.8184   0.8245 +0.0061 |    0.5294   0.5555 +0.0261 |   +0.0061
   fold7   10 |   0.8061   0.8166 +0.0105 |    0.5206   0.5298 +0.0092 |   +0.0105
   fold8   10 |   0.8128   0.8248 +0.0121 |    0.5288   0.5348 +0.0060 |   +0.0121
   fold9   10 |   0.8049   0.8125 +0.0075 |    0.5197   0.5384 +0.0187 |   +0.0075

  At val-selected epoch, across 10 folds:
    AUC :  S+L 0.8097±0.0047 | S+L+Z 0.8189±0.0046 | dZ +0.0092 wins 10/10
    AUPR:  S+L 0.5207±0.0107 | S+L+Z 0.5293±0.0148 | dZ +0.0087 wins 7/10

  early-stop vs run-to-end (dZ-AUC): selected +0.0092 | final-epoch +0.0092 | gain +0.0000

  PAIRED TEST  (S+L+Z) vs (S+L)  at val-selected epoch, 10 folds:
    AUC-ROC        mean dZ +0.0092 | wins 10/10 | paired-t t=+11.882 p=0.0000 | Wilcoxon W=0.0 p=0.0020 | dz=+3.76
    AUPR           mean dZ +0.0087 | wins 7/10 | paired-t t=+2.650 p=0.0265 | Wilcoxon W=7.0 p=0.0371 | dz=+0.84
    (dz = Cohen's d for paired diffs; |dz|>0.8 large. n folds is small,
     so Wilcoxon p floors at ~0.002 for n=10 / ~0.06 for n=5.)

[full-fit base] fit on ALL remainder (N=798, no CV, val included) -> score on FROZEN hold-out (N=199)

================  FULL-FIT BASE (all remainder, no CV)  ================
  fit N=798 (pos 174, rate 0.218) | test N=199 (pos 43, rate 0.216)
  S+L    :  AUC 0.8105  AUPR 0.5116
  S+L+Z  :  AUC 0.8211  AUPR 0.5225   (dZ AUC +0.0106  dZ AUPR +0.0108)


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

python debug_rl_rloo.py --holdout  --reps 10 --K 4 --rl_epochs 40   --eval_every 4 --clf cat --lr 1e-2 --stop_metric none --no_normalize      
