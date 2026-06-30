OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nohup python -u debug_rl_rloo.py \
  --holdout --reps 10 --K 4 --rl_epochs 80 --eval_every 4 \
  --clf tabpfn --lr 1e-3 --stop_metric AUC \
  > rloo_k10.log 2>&1 &
echo "PID $!"