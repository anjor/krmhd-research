#!/bin/bash
# Phase 1 Launch Script - Hermite Convergence Survey
# Usage: ./launch_phase1.sh

set -e  # Exit on error

echo "🔬 Problem 1 Phase 1: Hermite Convergence Survey"
echo "=================================================="
echo "Based on Anjor's direction (2026-02-22)"
echo ""

# Navigate to workspace
cd ~/.openclaw/workspace-physics

# Check GANDALF environment
echo "🔧 Checking GANDALF setup..."
if [ ! -d "gandalf" ]; then
    echo "❌ GANDALF directory not found. Please clone gandalf repo first."
    exit 1
fi

cd gandalf

# Check virtual environment
if [ ! -d ".venv" ]; then
    echo "🐍 Creating virtual environment..."
    uv venv
fi

# Activate and install dependencies
echo "📦 Installing dependencies..."
source .venv/bin/activate
uv sync

# Check JAX Metal backend
echo "🖥️  Checking JAX Metal backend..."
python -c "import jax; print('JAX devices:', jax.devices())" || {
    echo "❌ JAX Metal backend not working. Check installation."
    exit 1
}

echo "✅ GANDALF environment ready"
echo ""

# Return to workspace root
cd ..

# Launch Phase 1
echo "🚀 Launching Phase 1 simulations..."
python scripts/phase1_launch.py

echo ""
echo "🏁 Phase 1 launch completed!"
echo "📊 Monitor progress with: python scripts/monitor_phase1.py"