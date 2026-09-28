#!/usr/bin/env bash
#SBATCH --job-name=slic_si_neffall
#SBATCH --partition=day
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sineffall_%A_%a.out
#SBATCH --error=logs/sineffall_%A_%a.err

# Fig. 3 recomputed with n_eff (Reviewer 2 §3).
#   SLIC_SI_COND=ss  sbatch --array=0-35 cluster/submit_si_neff.sh   # subsampling
#   SLIC_SI_COND=Ts  sbatch --array=0-35 cluster/submit_si_neff.sh   # traj length
#   SLIC_SI_COND=dts sbatch --array=0-35 cluster/submit_si_neff.sh   # sampling freq
#   SLIC_SI_COND=ps  sbatch --array=0-23 cluster/submit_si_neff.sh   # params (4 sys)
# task = sys_idx*6 + noise_idx

set -euo pipefail
REPO_DIR="${REPO_DIR:-$(pwd)}"   # run from the repository root, or set REPO_DIR
source "${REPO_DIR}/cluster/env_setup.sh"
cd "${REPO_DIR}"

echo "=== Task ${SLURM_ARRAY_TASK_ID} | cond=${SLIC_SI_COND} ==="
echo "  Started: $(date)"
julia --project=. scripts/rev/rerun_si_neffall.jl
echo "  Finished: $(date)"
