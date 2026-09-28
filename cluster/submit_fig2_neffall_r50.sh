#!/usr/bin/env bash
#SBATCH --job-name=slic_fig2_r50
#SBATCH --partition=day
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=logs/fig2r50_%A_%a.out
#SBATCH --error=logs/fig2r50_%A_%a.err
# Fig. 2 neff_all at 50 runs.  sbatch --array=0-35 cluster/submit_fig2_neffall_r50.sh
# task = sys_idx*6 + noise_idx.  Walltime doubled vs the 25-run job (12h -> 23h).
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(pwd)}"   # run from the repository root, or set REPO_DIR
source "${REPO_DIR}/cluster/env_setup.sh"
cd "${REPO_DIR}"
echo "=== Task ${SLURM_ARRAY_TASK_ID} ===  $(date)"
julia --project=. scripts/rev/rerun_fig2_neffall_r50.jl
echo "  done $(date)"
