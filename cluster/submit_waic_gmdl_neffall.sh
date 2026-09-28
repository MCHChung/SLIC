#!/usr/bin/env bash
#SBATCH --job-name=slic_wg_neffall
#SBATCH --partition=day
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/wgna_%A_%a.out
#SBATCH --error=logs/wgna_%A_%a.err
# WAIC/gMDL under neff_all.  sbatch --array=0-35 cluster/submit_waic_gmdl_neffall.sh
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(pwd)}"   # run from the repository root, or set REPO_DIR
source "${REPO_DIR}/cluster/env_setup.sh"; cd "${REPO_DIR}"
echo "=== Task ${SLURM_ARRAY_TASK_ID} ===  $(date)"
julia --project=. scripts/rev/rerun_waic_gmdl_neffall.jl
echo "  done $(date)"
