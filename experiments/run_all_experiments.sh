#!/usr/bin/env bash
# Run all experiments sequentially and log output.
# Usage:  bash experiments/run_all_experiments.sh
set -euo pipefail

LOG_DIR="results"
mkdir -p "$LOG_DIR"

echo "===== Starting experiments: $(date) =====" | tee "$LOG_DIR/run.log"

echo -e "\n[1/2] D_r comparison (all variants vs OSPF)..." | tee -a "$LOG_DIR/run.log"
python -u experiments/comparison_experiment.py 2>&1 | tee -a "$LOG_DIR/run.log"

echo -e "\n[2/2] Topology robustness..." | tee -a "$LOG_DIR/run.log"
python -u experiments/topology_robustness_variants.py 2>&1 | tee -a "$LOG_DIR/run.log"

echo -e "\n===== All experiments done: $(date) =====" | tee -a "$LOG_DIR/run.log"
