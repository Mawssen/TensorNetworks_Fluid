#!/bin/bash
#SBATCH --job-name=mps_stats
#SBATCH --output=mps_stats_%A_%a.out
#SBATCH --error=mps_stats_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --array=0-3

# -----------------------------------------------------------------------------
# Run MPS stats for one (cutoff, sp) per array task. Cube only (MPS is 512^3).
#
# Variables (with defaults):
#   TIMESTEP=0198
#   Z_CHI_MODE=stored          # stored | mixed | recon
#   FD_ORDER=8                 # 2|4|6|8 (used only for mixed/recon)
#   PERIODIC_XZ=false          # true|false (used only for mixed/recon)
#   SP=1                       # 1|2|3|4 -> il|seq|comb1|combn ordering
#   DATA_DIR=truncated_<TS>_sp<SP>   (auto-derived; legacy sp=1 falls back
#                                    to truncated_<TS>/ if no _sp1 dir)
#
# Submit examples:
#   sbatch submit_mps_stats.sh                         # ts=0198, sp=1, stored
#   SP=2 sbatch submit_mps_stats.sh                    # sequential ordering
#   Z_CHI_MODE=recon SP=3 sbatch submit_mps_stats.sh   # comb1, recon mode
# -----------------------------------------------------------------------------

source /ix/pgivi/moe32/envs/my_env/bin/activate
module load texlive/2021
module load python/ondemand-jupyter-python3.11

TIMESTEP="${TIMESTEP:-0198}"
Z_CHI_MODE="${Z_CHI_MODE:-stored}"
FD_ORDER="${FD_ORDER:-8}"
PERIODIC_XZ="${PERIODIC_XZ:-false}"
SP="${SP:-1}"

# Locate input dir. Layout: truncated_<TS>/sp<SP>/  (sp folder INSIDE timestep folder)
# Legacy fallbacks: truncated_<TS>_sp<SP>/  or  truncated_<TS>/  (sp=1 only).
DEFAULT_DATA_DIR="$SLURM_SUBMIT_DIR/truncated_${TIMESTEP}/sp${SP}"
if [ ! -d "$DEFAULT_DATA_DIR" ]; then
    LEGACY_FLAT="$SLURM_SUBMIT_DIR/truncated_${TIMESTEP}_sp${SP}"
    LEGACY_NOSP="$SLURM_SUBMIT_DIR/truncated_${TIMESTEP}"
    if [ -d "$LEGACY_FLAT" ]; then
        DEFAULT_DATA_DIR="$LEGACY_FLAT"
    elif [ "$SP" = "1" ] && [ -d "$LEGACY_NOSP" ]; then
        DEFAULT_DATA_DIR="$LEGACY_NOSP"
    fi
fi
DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"

OUT_ROOT="${OUT_ROOT:-$SLURM_SUBMIT_DIR/results/mps_${TIMESTEP}_sp${SP}}"
RESULTS_ROOT="${RESULTS_ROOT:-$SLURM_SUBMIT_DIR/results}"

# Build a suffix for the comparison output dir based on what's varied.
# sp is always tagged; FD_ORDER and PERIODIC_XZ only for recon/mixed.
SUFFIX="${Z_CHI_MODE}_sp${SP}"
if [ "$Z_CHI_MODE" = "recon" ] || [ "$Z_CHI_MODE" = "mixed" ]; then
    SUFFIX="${Z_CHI_MODE}_o${FD_ORDER}_sp${SP}"
    if [ "$PERIODIC_XZ" = "true" ]; then
        SUFFIX="${SUFFIX}_p"
    fi
fi
COMPARISON_OUT="${COMPARISON_OUT:-$SLURM_SUBMIT_DIR/results/comparison_${TIMESTEP}_${SUFFIX}}"

CUTOFF_STRS=("0.01" "0.001" "0.0001" "1.0e-5")
CUTOFF_LABELS=("1e-2" "1e-3" "1e-4" "1e-5")

i=$SLURM_ARRAY_TASK_ID
CUTOFF_STR=${CUTOFF_STRS[$i]}
CUTOFF_LABEL=${CUTOFF_LABELS[$i]}
OUT_DIR="$OUT_ROOT/$CUTOFF_LABEL"
mkdir -p "$OUT_DIR"

echo "TIMESTEP    = $TIMESTEP"
echo "DATA_DIR    = $DATA_DIR"
echo "CUTOFF_STR  = $CUTOFF_STR ($CUTOFF_LABEL)"
echo "Z_CHI_MODE  = $Z_CHI_MODE"
echo "FD_ORDER    = $FD_ORDER"
echo "PERIODIC_XZ = $PERIODIC_XZ"
echo "SP          = $SP"
echo "OUT_DIR     = $OUT_DIR"
which python

python run_mps_stats.py "$DATA_DIR" "$TIMESTEP" "$CUTOFF_STR" "$OUT_DIR" \
    --z-chi-mode "$Z_CHI_MODE" --fd-order "$FD_ORDER" \
    --periodic-xz "$PERIODIC_XZ" --sp "$SP"
RC=$?
if [ $RC -ne 0 ]; then
    echo "FAILED rc=$RC"
    exit $RC
fi

# -----------------------------------------------------------------------------
# Submit the comparison plotter as a follow-up (depends on full array success).
# -----------------------------------------------------------------------------
LOCK_FILE="$OUT_ROOT/.comparison_submitted_${SLURM_ARRAY_JOB_ID}"
if ( set -o noclobber; > "$LOCK_FILE" ) 2>/dev/null; then
    echo "[$SLURM_ARRAY_TASK_ID] Submitting comparison plot job"
    if [ "$Z_CHI_MODE" = "recon" ] || [ "$Z_CHI_MODE" = "mixed" ]; then
        FD_FLAG="--fd-order ${FD_ORDER}"
        # Apples-to-apples: tell plotter to recompute DNS chi with the
        # same operator used by MPS reconstruction.
        DNS_DATA_FLAG="--dns-data-dir ${SLURM_SUBMIT_DIR}/jet_${TIMESTEP}"
    else
        FD_FLAG=""
        DNS_DATA_FLAG=""
    fi
    PERIODIC_FLAG="--periodic-xz ${PERIODIC_XZ}"
    SP_FLAG="--sp ${SP}"
    sbatch \
        --job-name=mps_compare \
        --output="${SLURM_SUBMIT_DIR}/mps_compare_%j.out" \
        --error="${SLURM_SUBMIT_DIR}/mps_compare_%j.err" \
        --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 \
        --cluster=smp --partition=smp \
        --mem=32G --time=0:45:00 \
        --dependency=afterok:${SLURM_ARRAY_JOB_ID} \
        --wrap="
source /ix/pgivi/moe32/envs/my_env/bin/activate
module load texlive/2021
module load python/ondemand-jupyter-python3.11
cd ${SLURM_SUBMIT_DIR}
echo 'Generating DNS-vs-MPS comparison plots for timestep ${TIMESTEP}, sp=${SP}...'
python plot_dns_mps_comparison.py \"${RESULTS_ROOT}\" \"${COMPARISON_OUT}\" --timestep ${TIMESTEP} --mode ${Z_CHI_MODE} ${FD_FLAG} ${PERIODIC_FLAG} ${SP_FLAG} ${DNS_DATA_FLAG}
"
else
    echo "[$SLURM_ARRAY_TASK_ID] Comparison job already submitted by another task."
fi

echo "Done."
