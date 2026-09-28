#!/usr/bin/env bash
#SBATCH --job-name=slic_lt_allsys
#SBATCH --partition=week
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/ltall_%A_%a.out
#SBATCH --error=logs/ltall_%A_%a.err
# Long-trajectory, ALL 6 systems x 6 L values = 36 tasks.
# task = s_idx*6 + L_idx  (s_idx 0..5 = sys 1..6 ; L_idx 0..5 = L 1,2,5,10,20,50)
# Runs are reduced automatically at the expensive cells: L>=50 -> 8 runs,
# L>=20 -> 15 runs, otherwise 25.
#   short L only:  sbatch --array=0-3,6-9,12-15,18-21,24-27,30-33 ...
#   long  L only:  sbatch --array=4,5,10,11,16,17,22,23,28,29,34,35 ...
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(pwd)}"   # run from the repository root, or set REPO_DIR
source "${REPO_DIR}/cluster/env_setup.sh"; cd "${REPO_DIR}"
echo "=== Task ${SLURM_ARRAY_TASK_ID} ===  $(date)"
julia --project=. scripts/rev/rerun_longtraj_allsys.jl
echo "  done $(date)"
