#!/usr/bin/env bash
# Usage:
#   ./run_grid.sh                                        # random-init ScaIR
#   ./run_grid.sh checkpoints_delay/episode_0400.pt      # trained checkpoint
#   ./run_grid.sh checkpoints_delay_opt2/episode_0400.pt --delay_input
#   ./run_grid.sh checkpoints_delay_opt3/episode_0400.pt --delay_init
set -e

CHECKPOINT="${1:-}"
EXTRA_FLAGS="${@:2}"   # any extra flags (--delay_input, --delay_init, ...)

TOPO="data/ABI/Topology.txt"
TM_DIR="data/ABI/TrafficMatrix"
LINK_WEIGHTS="data/ABI/link_weight.json"

CKPT_ARG=""
if [ -n "$CHECKPOINT" ]; then
    CKPT_ARG="--checkpoint $CHECKPOINT"
fi

python grid_eval.py \
    --topo         "$TOPO" \
    --tm_dir       "$TM_DIR" \
    --link_weights "$LINK_WEIGHTS" \
    $CKPT_ARG \
    --dr_values    0.1 0.2 0.3 0.5 0.7 0.9 \
    --packet_counts 50 100 200 \
    --episodes     50 \
    --save_plot    results/grid_eval.png \
    $EXTRA_FLAGS
