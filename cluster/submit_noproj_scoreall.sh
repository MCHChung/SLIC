#!/usr/bin/env bash
#SBATCH --job-name=slic_noproj_scoreall
#SBATCH --partition=day
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/noproj_sa_%A_%a.out
#SBATCH --error=logs/noproj_sa_%A_%a.err

# ============================================================================
# No-projection diagnostic across ALL 6 systems (CalcDeriv, 4th-order FD).
#
# Task = sys - 1:
#   0=Lorenz, 1=Rossler, 2=LV, 3=Brusselator, 4=VdP, 5=Pendulum
#
# First-order systems (0,1,2,3): one 'fully_observed' condition, single FD.
# Second-order systems (4,5): two conditions ('estimate_v' double FD,
#                             'known_v' single FD).
#
# Submit all six:
#   sbatch --array=0-5 cluster/submit_no_projection_all.sh
#
# Each task sweeps all noise levels internally; ~10-40 min per system.
# ============================================================================

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"   # run from the repository root, or set REPO_DIR
source "${REPO_DIR}/cluster/env_setup.sh"

cd "${REPO_DIR}"

echo "=== Task ${SLURM_ARRAY_TASK_ID} (no-projection) ==="
echo "  Started: $(date)"
julia --project=. scripts/rev/rerun_noproj_scoreall.jl
echo "  Finished: $(date)"
