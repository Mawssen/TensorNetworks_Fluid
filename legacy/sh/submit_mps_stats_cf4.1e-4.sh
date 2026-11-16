#!/bin/bash
#SBATCH --job-name=mps_stats_cf4.1e-4
#SBATCH --output=mps_stats_cf4.1e-4_%j.out
#SBATCH --error=mps_stats_cf4.1e-4_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --mem=64G
#SBATCH --time=1:00:00

# -----------------------------------------------------------------------------
# One-shot SLURM job: compute MPS stats for the cf=4.1e-4 case
# (LES-matched cutoff variant; pairs with chi=93).
# Assumes 7 fields are already truncated at cf=4.1e-4:
#   truncated_0198/sp1/truncated_jet_<var>_0198.dat_512_il_cf4.1e-4.mat
# for var in mixfrac T Y_CO Y_CO2 Y_OH Y_O2 alpha
#
# Submit:
#   sbatch submit_mps_stats_cf4.1e-4.sh
# -----------------------------------------------------------------------------

source /ix/pgivi/moe32/envs/my_env/bin/activate
module load texlive/2021
module load python/ondemand-jupyter-python3.11

TIMESTEP="${TIMESTEP:-0198}"
SP="${SP:-1}"
Z_CHI_MODE="${Z_CHI_MODE:-mixed}"
FD_ORDER="${FD_ORDER:-4}"
PERIODIC_XZ="${PERIODIC_XZ:-false}"

DEFAULT_DATA_DIR="$SLURM_SUBMIT_DIR/truncated_${TIMESTEP}/sp${SP}"
if [ ! -d "$DEFAULT_DATA_DIR" ]; then
    LEGACY_FLAT="$SLURM_SUBMIT_DIR/truncated_${TIMESTEP}_sp${SP}"
    LEGACY_NOSP="$SLURM_SUBMIT_DIR/truncated_${TIMESTEP}"
    if [ -d "$LEGACY_FLAT" ]; then DEFAULT_DATA_DIR="$LEGACY_FLAT"
    elif [ "$SP" = "1" ] && [ -d "$LEGACY_NOSP" ]; then DEFAULT_DATA_DIR="$LEGACY_NOSP"
    fi
fi
DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"

OUT_ROOT="${OUT_ROOT:-$SLURM_SUBMIT_DIR/results/mps_${TIMESTEP}_sp${SP}}"
# The cutoff string used IN THE FILENAME. The Julia script writes
# truncated_jet_<var>_0198.dat_512_il_cf4.1e-4.mat, so this must match.
CUTOFF_STR="0.00041"
CUTOFF_LABEL="4.1e-4"
OUT_DIR="$OUT_ROOT/$CUTOFF_LABEL"
mkdir -p "$OUT_DIR"

echo "TIMESTEP    = $TIMESTEP"
echo "DATA_DIR    = $DATA_DIR"
echo "CUTOFF_STR  = $CUTOFF_STR"
echo "Z_CHI_MODE  = $Z_CHI_MODE"
echo "FD_ORDER    = $FD_ORDER"
echo "PERIODIC_XZ = $PERIODIC_XZ"
echo "SP          = $SP"
echo "OUT_DIR     = $OUT_DIR"

python run_mps_stats.py "$DATA_DIR" "$TIMESTEP" "$CUTOFF_STR" "$OUT_DIR" \
    --z-chi-mode "$Z_CHI_MODE" --fd-order "$FD_ORDER" \
    --periodic-xz "$PERIODIC_XZ" --sp "$SP"
RC=$?
echo "Done with rc=$RC"
exit $RC
