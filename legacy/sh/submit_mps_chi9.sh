#!/bin/bash
#SBATCH --job-name=mps_chi9_uvw
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=3:00:00
#SBATCH --output=mps_chi9_uvw_%j.out

set -euo pipefail

# --- Reproduce the interactive environment (juliaup on PATH etc.) ----------
# Batch shells don't source ~/.bashrc; relax set -e/-u around it.
set +eu
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
set -eu

# The interactive Julia that HAS ITensorMPS is juliaup's 1.12, NOT a system
# `module load julia`. Call it directly.
JULIA="$HOME/.juliaup/bin/julia"
echo "Using Julia: $JULIA"; "$JULIA" --version

# Private per-job compile cache (avoids parallel-precompile corruption if you
# ever run several of these at once); reads installed packages from ~/.julia.
JOBCACHE="${SLURM_SCRATCH:-$HOME/.julia_jobcache/$SLURM_JOB_ID}/julia_depot"
mkdir -p "$JOBCACHE"
export JULIA_DEPOT_PATH="$JOBCACHE:$HOME/.julia"

# ---- edit these to match your setup --------------------------------------
# Directory containing mps_calc.jl (run from here so its relative paths work).
RUNDIR=/ix/pgivi/moe32/Schmidt/Forced_Isotropic/5_Grid_1024/Codes/calculations/1Generate_truncated_velocity/1U_e2/cf_sweep
# Directory holding the jet velocity component files.
DATADIR=/ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198
# Timestep / file stems for the three components.
TS=0198
COMPONENTS=(jet_u_${TS} jet_v_${TS} jet_w_${TS})
CHI=9
SPLIT=1
# --------------------------------------------------------------------------

cd "$RUNDIR"

for comp in "${COMPONENTS[@]}"; do
    echo "=================================================================="
    echo "[mps chi=$CHI] component file = ${comp}"
    echo "=================================================================="
    "$JULIA" mps_calc.jl \
        chi=${CHI} \
        cutoff=0.0 \
        split=${SPLIT} \
        file=${comp} \
        dir=${DATADIR} \
      || echo "[mps chi=$CHI] FAILED for ${comp} (continuing)"
    echo ""
done

echo "[mps chi=$CHI] all components done. Check mps_calc.jl's output .mat files."
