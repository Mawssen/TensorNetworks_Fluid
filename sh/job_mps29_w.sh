#!/bin/bash
#SBATCH --job-name=mps29_w
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=3:00:00
#SBATCH --output=/ix/pgivi/moe32/Aidyn_DNS/stats/mps_chi29_jobs/mps29_w_%j.out

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

cd /ix/pgivi/moe32/Aidyn_DNS/stats

echo "[mps chi=29] component file = jet_w_0198"
"$JULIA" mps_calc.jl \
    chi=29 \
    cutoff=0.0 \
    split=1 \
    file=jet_w_0198 \
    dir=/ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198 \
    ext=dat

echo "[mps chi=29] jet_w_0198 done."
