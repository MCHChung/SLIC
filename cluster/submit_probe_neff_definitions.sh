#!/usr/bin/env bash
#SBATCH --job-name=slic_neffdef
#SBATCH --partition=day
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/neffdef_%A_%a.out
#SBATCH --error=logs/neffdef_%A_%a.err
# n_eff definition robustness.  sbatch --array=0-5 cluster/submit_probe_neff_definitions.sh
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(pwd)}"   # run from the repository root, or set REPO_DIR
source "${REPO_DIR}/cluster/env_setup.sh"; cd "${REPO_DIR}"
echo "=== case ${SLURM_ARRAY_TASK_ID} ===  $(date)"
julia --project=. scripts/rev/probe_neff_definitions.jl
echo "  done $(date)"
