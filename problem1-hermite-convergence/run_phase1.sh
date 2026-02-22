#!/bin/bash
# Phase 1 Launcher - runs all 9 configurations sequentially
set -e

GANDALF_DIR=~/.openclaw/workspace-physics/gandalf
CONFIGS_DIR=~/.openclaw/workspace-physics/krmhd-research/problem1-hermite-convergence/configs
RESULTS_DIR=~/.openclaw/workspace-physics/krmhd-research/problem1-hermite-convergence/results

cd $GANDALF_DIR
source .venv/bin/activate

mkdir -p $RESULTS_DIR
mkdir -p logs

echo "🔬 Phase 1: Hermite Convergence Study"
echo "======================================"
echo "Configs: $CONFIGS_DIR"
echo ""

for config in $CONFIGS_DIR/phase1_*.yaml; do
    name=$(basename $config .yaml)
    echo "🚀 Starting: $name"
    
    # Run simulation, save output to log
    python scripts/run_simulation.py "$config" \
        --output-dir "$RESULTS_DIR/$name" \
        2>&1 | tee "logs/${name}.log"
    
    echo "✅ Completed: $name"
    echo ""
done

echo "🏁 Phase 1 complete!"
