#!/usr/bin/env bash
# Evaluate each method at its BEST setting across 10 independent hold-out seeds.
# Each seed = a disjoint frozen hold-out -> one independent dZ point -> the
# cross-seed paired test (n=10) at the end of each run is the valid significance
# test (see print_cross_seed; n>=10 unlocks the p-values).
#
# Best settings (fixed):
#   xgb    : lr 1e-2, rl_epochs 40, eval_every 4
#   cat    : lr 1e-2, rl_epochs 40, eval_every 4
#   tabpfn : lr 1e-3, rl_epochs 10, eval_every 2
#
# All output -> screen AND log_sweep_seeds.txt (one file, appended).

set -u
LOG="log_sweep_seeds.txt"
: > "$LOG"

# 10 independent hold-out seeds
SEEDS="12345 111 222 333 444 555 666 777 888 999"

run() {
  # $1=clf  $2=lr  $3=epochs  $4=eval
  local clf="$1" lr="$2" ep="$3" ev="$4"
  local header="######## clf=${clf} lr=${lr} rl_epochs=${ep} eval_every=${ev} | 10 hold-out seeds ########"
  printf '\n\n%s\n' "$header" | tee -a "$LOG"
  python debug_rl_rloo.py --holdout \
      --holdout_seeds $SEEDS \
      --reps 10 --K 4 \
      --rl_epochs "$ep" --eval_every "$ev" \
      --clf "$clf" --lr "$lr" \
      --stop_metric none --no_normalize --neutral_init \
      --log none 2>&1 | tee -a "$LOG"
}

run xgb    1e-2 40 4
run cat    1e-2 40 4
run tabpfn 1e-3 10 2

printf '\n\n######## seed sweep done -> %s ########\n' "$LOG" | tee -a "$LOG"