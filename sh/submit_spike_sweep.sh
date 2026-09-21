#!/bin/bash
#SBATCH --job-name=mps_spike_sweep
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=mps_spike_sweep_%j.out

set -euo pipefail

source /ix/pgivi/moe32/envs/my_env/bin/activate
module load python/ondemand-jupyter-python3.11

STATS=/ix/pgivi/moe32/Aidyn_DNS/stats
TS=0198
OUTROOT=$STATS/QC4PDE/spike_sweep
mkdir -p "$OUTROOT"

# ---- edit these two lists to control the sweep -----------------------------
# CUTOFFS: use the labels/tokens that appear in your MPS filenames.
#   chi93 is the fixed-bond-dimension case; the others are singular-value
#   cutoffs. Pass whatever tokens your _mps_filename expects (the script
#   passes unknown values through verbatim).
CUTOFFS=("0.001" "0.0001" "1.0e-5")
# SPS: the sp (ordering / stride) values you want to check.
SPS=(1 2)
# n-sigma: seam-detection threshold. Raise (e.g. 6) if a run flags >30% of
# planes (the script prints a WARNING when that happens).
NSIGMA=6.0
# ---------------------------------------------------------------------------

for sp in "${SPS[@]}"; do
  for cf in "${CUTOFFS[@]}"; do
    tag="cf${cf}_sp${sp}"
    echo "=================================================================="
    echo "[sweep] cutoff=${cf}  sp=${sp}   (n-sigma=${NSIGMA})"
    echo "=================================================================="
    python diagnose_mps_spikes.py \
        --dns-data-dir "$STATS/jet_${TS}" \
        --mps-root     "$STATS" \
        --timestep     "$TS" \
        --sp           "$sp" \
        --cutoff       "$cf" \
        --n-sigma      "$NSIGMA" \
        --out          "$OUTROOT/spike_${tag}.png" \
      || echo "[sweep] FAILED for ${tag} (continuing)"
    echo ""
  done
done

echo "[sweep] all done. Figures in $OUTROOT/ ; full console log in the .out file."
