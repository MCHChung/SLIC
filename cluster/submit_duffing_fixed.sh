#!/usr/bin/env bash
#SBATCH --job-name=slic_duff_fix
#SBATCH --partition=day
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/dufffix_%A_%a.out
#SBATCH --error=logs/dufffix_%A_%a.err
# Duffing sweep with a FIXED initial condition.
#   sbatch --array=0-8 cluster/submit_duffing_fixed.sh
# One task per alpha in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0].
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(pwd)}"   # run from the repository root, or set REPO_DIR
source "${REPO_DIR}/cluster/env_setup.sh"
cd "${REPO_DIR}"
echo "=== alpha index ${SLURM_ARRAY_TASK_ID} ===  $(date)"
julia --project=. scripts/rev/rerun_duffing_fixed.jl
echo "  done $(date)"
