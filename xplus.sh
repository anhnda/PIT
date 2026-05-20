#!/bin/bash

# --- GRUD+ Experiments ---
echo "========================================="
echo " STARTING GRUD+ EXPERIMENTS"
echo "========================================="

echo -e "\n--- [1/3] Running GRUD+ with TabPFN ---"
python grud_plus.py 2>&1 | tee -a grud_plus.log

echo -e "\n--- [2/3] Running GRUD+ with XGBoost ---"
python grud_plus.py --clf xgboost 2>&1 | tee -a grud_plus.log

echo -e "\n--- [3/3] Running GRUD+ with CatBoost ---"
python grud_plus.py --clf catboost 2>&1 | tee -a grud_plus.log


# --- ODE+ Experiments ---
echo -e "\n========================================="
echo " STARTING ODE+ EXPERIMENTS"
echo "========================================="

echo -e "\n--- [1/3] Running ODE+ with TabPFN ---"
python ode_plus.py 2>&1 | tee -a ode_plus.log

echo -e "\n--- [2/3] Running ODE+ with XGBoost ---"
python ode_plus.py --clf xgboost 2>&1 | tee -a ode_plus.log

echo -e "\n--- [3/3] Running ODE+ with CatBoost ---"
python ode_plus.py --clf catboost 2>&1 | tee -a ode_plus.log

echo -e "\nAll experiments finished!"