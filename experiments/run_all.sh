#!/usr/bin/env bash
# Run all overnight experiments sequentially.
# Outputs go to results/  Logs go to results/run_all.log

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOGFILE="results/run_all.log"
mkdir -p results

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGFILE"; }

log "=== Starting all experiments ==="

# 1. Topology robustness
log "--- Experiment 1: Topology Robustness ---"
python experiments/topology_robustness.py 2>&1 | tee -a "$LOGFILE"
log "--- Topology Robustness done ---"

# 2. Shared SubGNN
log "--- Experiment 2: Shared SubGNN ---"
python experiments/shared_gnn_experiment.py 2>&1 | tee -a "$LOGFILE"
log "--- Shared SubGNN done ---"

log "=== All experiments complete. Results in results/ ==="
ls -lh results/
