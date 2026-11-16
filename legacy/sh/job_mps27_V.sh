#!/bin/bash
#SBATCH --job-name=mps27_V
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=3:00:00
#SBATCH --output=/ix/pgivi/moe32/Aidyn_DNS/stats/mps_chi27_jobs/mps27_V_%j.out

set -euo pipefail

set +eu
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
set -eu

JULIA="$HOME/.juliaup/bin/julia"
echo "Using Julia: $JULIA"; "$JULIA" --version

JOBCACHE="${SLURM_SCRATCH:-$HOME/.julia_jobcache/$SLURM_JOB_ID}/julia_depot"
mkdir -p "$JOBCACHE"
export JULIA_DEPOT_PATH="$JOBCACHE:$HOME/.julia"

cd /ix/pgivi/moe32/Schmidt/Forced_Isotropic/5_Grid_1024/Codes/calculations/1Generate_truncated_velocity/1U_e2/cf_sweep

echo "[mps chi=27] component file = V_t1.mat"
"$JULIA" mps_calc.jl \
    chi=27 \
    cutoff=0.0 \
    split=1 \
    file=V_t1 \
    dir=/ix/pgivi/moe32/Schmidt/Forced_Isotropic/5_Grid_1024/Forced_Isotropic_1024Cubed_mat_in_one \
    ext=mat

echo "[mps chi=27] V_t1 done."
