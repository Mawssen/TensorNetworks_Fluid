#!/bin/bash
#SBATCH --job-name=les_plane_diag
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:45:00
#SBATCH --output=les_plane_diag_%j.out

set -euo pipefail

source /ix/pgivi/moe32/envs/my_env/bin/activate
module load python/ondemand-jupyter-python3.11

STATS=/ix/pgivi/moe32/Aidyn_DNS/stats
TS=0198

python diagnose_les_plane.py \
    --dns-data-dir "$STATS/jet_${TS}" \
    --les-plt      "$STATS/jet_${TS}/LES_plt20000" \
    --timestep     "$TS" \
    --out          "$STATS/QC4PDE/les_plane_diag.png"
