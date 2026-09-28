#!/usr/bin/env bash
# Supplementary Fig. S12 (final run). First compute the noise-free residuals of
# the true models (once per system):
#   sbatch --array=0-5 cluster/submit_calibrate_c.sh
# then run the enumeration with the residual floor gamma = 100 for each system:
#   SLIC_GAMMA=100 SLIC_ENUM_SYS=3 sbatch --array=0-11 cluster/submit_enumerate_small.sh   # Lotka-Volterra
#   SLIC_GAMMA=100 SLIC_ENUM_SYS=4 sbatch --array=0-11 cluster/submit_enumerate_small.sh   # Brusselator
#   SLIC_GAMMA=100 SLIC_ENUM_SYS=5 sbatch --array=0-11 cluster/submit_enumerate_small.sh   # Van der Pol
#   SLIC_GAMMA=100 SLIC_ENUM_SYS=6 sbatch --array=0-5  cluster/submit_enumerate_small.sh   # nonlinear pendulum
# Task id = (noise level - 1) * (number of equations) + (equation - 1).
# Results: data/sims/ode_results_rev/enumeration_g100.0/<system>/
#SBATCH --job-name=slic_enum_small
#SBATCH --partition=day
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/enum_small_%A_%a.out
#SBATCH --error=logs/enum_small_%A_%a.err

# ============================================================================
# Enumeration for LV (sys=3) and Brusselator (sys=4).
# Both have 2-state systems with small libraries, so 6 noise × 2 eq = 12 tasks
# per system. Submit twice with SLIC_ENUM_SYS set to 3 then 4.
#
# Usage:
#   SLIC_ENUM_SYS=3 sbatch --array=0-11 cluster/submit_enumerate_small.sh
#   SLIC_ENUM_SYS=4 sbatch --array=0-11 cluster/submit_enumerate_small.sh
# ============================================================================

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"   # run from the repository root, or set REPO_DIR
source "${REPO_DIR}/cluster/env_setup.sh"

cd "${REPO_DIR}"

: "${SLIC_ENUM_SYS:?Must set SLIC_ENUM_SYS (3 or 4) when submitting}"
export SLIC_ENUM_SYS

echo "=== Task ${SLURM_ARRAY_TASK_ID} (sys=${SLIC_ENUM_SYS}) ==="
echo "  Started: $(date)"
julia --project=. scripts/rev/enumerate_all.jl
echo "  Finished: $(date)"
