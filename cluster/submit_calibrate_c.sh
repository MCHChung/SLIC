#!/usr/bin/env bash
#SBATCH --job-name=slic_ccal
#SBATCH --partition=day
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ccal_%A_%a.out
#SBATCH --error=logs/ccal_%A_%a.err

# ============================================================================
# Calibrate the conditioning weight c on noise-free data, one task per system.
#
# Task = sys - 1:
#   0=Lorenz  1=Rossler  2=LV  3=Brusselator  4=VdP  5=NLP
#
# Submit all six:
#   sbatch --array=0-5 cluster/submit_calibrate_c.sh
#
# Each task sweeps the c grid at noise=0 and prints, per criterion:
#   E/·  = exact recovery of the true support (all equations)
#   digits = number of terms selected, one digit per equation
#   η/RSS = conditioning floor relative to the systematic residual
#
# Read the calibrated c off the "<== ALL EXACT" rows: take the smallest, and
# check the window extends above it before under-selection sets in.
#
# Optional: override the grid, e.g.
#   SLIC_CAL_CGRID=0.01,0.03,0.1,0.3,1.0 sbatch --array=0-5 cluster/submit_calibrate_c.sh
# ============================================================================

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"   # run from the repository root, or set REPO_DIR
source "${REPO_DIR}/cluster/env_setup.sh"

cd "${REPO_DIR}"

export SLIC_CAL_SYS=$(( SLURM_ARRAY_TASK_ID + 1 ))

echo "=== c-calibration: sys=${SLIC_CAL_SYS} ==="
echo "  Started: $(date)"
julia --project=. scripts/rev/calibrate_c.jl
echo "  Finished: $(date)"
