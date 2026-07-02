#!/usr/bin/env bash
# Sweep clf x lr for debug_rl_rloo.py --holdout --neutral_init.
#   clf: tabpfn, xgb, cat      lr: 1e-2, 1e-3, 1e-4
#   tabpfn -> rl_epochs 10, eval_every 2 ; others -> rl_epochs 40, eval_every 4
# All output goes to BOTH the screen and log_sweep_rl.txt (appended, one file).

set -u
LOG="log_sweep_rl.txt"
: > "$LOG"                      # truncate once at start; every run appends below

CLFS=(tabpfn xgb cat)
LRS=(1e-2 1e-3 1e-4)

for clf in "${CLFS[@]}"; do
  if [ "$clf" = "tabpfn" ]; then
    EPOCHS=10; EVAL=2
  else
    EPOCHS=40; EVAL=4
  fi
  for lr in "${LRS[@]}"; do
    header="######## clf=${clf} lr=${lr} rl_epochs=${EPOCHS} eval_every=${EVAL} ########"
    # print banner to screen + file
    printf '\n\n%s\n' "$header" | tee -a "$LOG"

    # run; stdout+stderr -> screen and appended to the shared log
    python debug_rl_rloo.py --holdout --reps 10 --K 4 \
        --rl_epochs "$EPOCHS" --eval_every "$EVAL" \
        --clf "$clf" --lr "$lr" \
        --stop_metric none --no_normalize --neutral_init \
        --log none 2>&1 | tee -a "$LOG"
  done
done

printf '\n\n######## sweep done -> %s ########\n' "$LOG" | tee -a "$LOG"