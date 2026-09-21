#!/bin/bash
#SBATCH --job-name=les_stats
#SBATCH --output=les_stats_%j.out
#SBATCH --error=les_stats_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --mem=32G
#SBATCH --time=1:00:00

# -----------------------------------------------------------------------------
# Compute LES statistics pickle for one plt file.
#
# Variables (with defaults):
#   TIMESTEP=0198
#   PLT_PATH=$SLURM_SUBMIT_DIR/jet_<TS>/LES_plt20000
#   Z_CHI_MODE=stored           # stored | mixed | recon
#   FD_ORDER=4                  # used only for mixed/recon
#   PERIODIC_XZ=false
#   OUT_DIR=$SLURM_SUBMIT_DIR/results/les_<TS>
#
# Submit:
#   sbatch submit_les_stats.sh
#   TIMESTEP=0198 Z_CHI_MODE=mixed sbatch submit_les_stats.sh
# -----------------------------------------------------------------------------

source /ix/pgivi/moe32/envs/my_env/bin/activate
module load texlive/2021
module load python/ondemand-jupyter-python3.11

TIMESTEP="${TIMESTEP:-0198}"
PLT_PATH="${PLT_PATH:-$SLURM_SUBMIT_DIR/jet_${TIMESTEP}/LES_plt20000}"
Z_CHI_MODE="${Z_CHI_MODE:-stored}"
FD_ORDER="${FD_ORDER:-4}"
PERIODIC_XZ="${PERIODIC_XZ:-false}"
OUT_DIR="${OUT_DIR:-$SLURM_SUBMIT_DIR/results/les_${TIMESTEP}}"

mkdir -p "$OUT_DIR"

echo "TIMESTEP    = $TIMESTEP"
echo "PLT_PATH    = $PLT_PATH"
echo "Z_CHI_MODE  = $Z_CHI_MODE"
echo "FD_ORDER    = $FD_ORDER"
echo "PERIODIC_XZ = $PERIODIC_XZ"
echo "OUT_DIR     = $OUT_DIR"

python run_les_stats.py "$PLT_PATH" "$OUT_DIR" \
    --timestep "$TIMESTEP" \
    --z-chi-mode "$Z_CHI_MODE" \
    --fd-order "$FD_ORDER" \
    --periodic-xz "$PERIODIC_XZ"

echo "Done."
