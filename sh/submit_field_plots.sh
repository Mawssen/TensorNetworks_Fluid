#!/bin/bash
# -----------------------------------------------------------------------------
# Submit field-by-field plotting jobs (scatter, PDF, contour) for the
# specified field(s), in parallel where possible.
#
# Layout:
#   - Scatter: one SLURM array task per (field, sp, cf). Each task is one plot.
#     Default: 3 fields x 4 sps x 4 cfs = 48 array tasks.
#     Each task ~5-10 min wall (mostly DNS/MPS field load).
#   - PDF:    one task per field (handles all sp+cf overlays internally).
#     Default: 3 tasks.
#   - Contour: one task per field. Default: 3 tasks.
#
# Customize via env vars:
#   TIMESTEP        (default 0198)
#   FIELDS          (default "mixfrac T chi")
#   SPS             (default "1 2 3 4")
#   CFS             (default "1e-2 1e-3 1e-4 1e-5")
#   MODE            (default "mixed"; only matters for chi)
#   FD_ORDER        (default 8)
#   PERIODIC_XZ     (default false)
#   PLOT_TYPES      (default "scatter pdf contour"; subset to skip types)
# -----------------------------------------------------------------------------

set -euo pipefail

TIMESTEP="${TIMESTEP:-0198}"
FIELDS="${FIELDS:-mixfrac T chi}"
SPS="${SPS:-1 2 3 4}"
CFS="${CFS:-1e-2 1e-3 1e-4 1e-5}"
MODE="${MODE:-mixed}"
FD_ORDER="${FD_ORDER:-4}"
PERIODIC_XZ="${PERIODIC_XZ:-false}"
PLOT_TYPES="${PLOT_TYPES:-scatter pdf contour}"

DNS_DATA_DIR="${DNS_DATA_DIR:-${PWD}/jet_${TIMESTEP}}"
MPS_ROOT="${MPS_ROOT:-${PWD}}"   # parent dir containing truncated_<TS>_sp<N>/
# Optional LES plt path; if a directory exists, contours and PDFs get an LES overlay.
LES_PLT="${LES_PLT:-${DNS_DATA_DIR}/LES_plt20000}"
OUT_BASE="${OUT_BASE:-${PWD}/results/plots_field_${TIMESTEP}}"
SCATTER_OUT="${OUT_BASE}/scatter"
PDF_OUT="${OUT_BASE}/pdf"
CONTOUR_OUT="${OUT_BASE}/contour"
mkdir -p "$SCATTER_OUT" "$PDF_OUT" "$CONTOUR_OUT"

echo "Submitting field plots:"
echo "  TIMESTEP    = $TIMESTEP"
echo "  FIELDS      = $FIELDS"
echo "  SPS         = $SPS"
echo "  CFS         = $CFS"
echo "  MODE        = $MODE"
echo "  FD_ORDER    = $FD_ORDER"
echo "  PERIODIC_XZ = $PERIODIC_XZ"
echo "  PLOT_TYPES  = $PLOT_TYPES"
echo "  DNS_DATA    = $DNS_DATA_DIR"
echo "  MPS_ROOT    = $MPS_ROOT"
echo "  OUT_BASE    = $OUT_BASE"
echo

n_jobs=0
SCATTER_JIDS=()

# ---- SCATTER: one job per (field, sp, cf) ----
if [[ " $PLOT_TYPES " == *" scatter "* ]]; then
    for FIELD in $FIELDS; do
        for SP in $SPS; do
            MPS_DIR="${MPS_ROOT}/truncated_${TIMESTEP}/sp${SP}"
            if [ ! -d "$MPS_DIR" ]; then
                if [ -d "${MPS_ROOT}/truncated_${TIMESTEP}_sp${SP}" ]; then
                    MPS_DIR="${MPS_ROOT}/truncated_${TIMESTEP}_sp${SP}"
                elif [ "$SP" = "1" ] && [ -d "${MPS_ROOT}/truncated_${TIMESTEP}" ]; then
                    MPS_DIR="${MPS_ROOT}/truncated_${TIMESTEP}"
                else
                    echo "  [skip] scatter sp=$SP: $MPS_DIR not found"
                    continue
                fi
            fi
            for CF in $CFS; do
                n_jobs=$((n_jobs + 1))
                JOB_NAME="sc_${FIELD}_sp${SP}_${CF}"
                JID=$(sbatch --parsable \
                    --job-name="$JOB_NAME" \
                    --output="${PWD}/logs_field/${JOB_NAME}_%j.out" \
                    --error="${PWD}/logs_field/${JOB_NAME}_%j.err" \
                    --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 \
                    --cluster=smp --partition=smp \
                    --mem=24G --time=0:30:00 \
                    --wrap="
mkdir -p ${PWD}/logs_field
source /ix/pgivi/moe32/envs/my_env/bin/activate
module load python/ondemand-jupyter-python3.11
cd ${PWD}
python plot_field_scatter.py \
    --field $FIELD --sp $SP --cf $CF \
    --timestep $TIMESTEP \
    --dns-data-dir $DNS_DATA_DIR \
    --mps-data-dir $MPS_DIR \
    --out-dir $SCATTER_OUT \
    --mode $MODE --fd-order $FD_ORDER --periodic-xz $PERIODIC_XZ
")
                # parsable returns "JOBID;CLUSTER" -> keep just the JID
                JID="${JID%%;*}"
                SCATTER_JIDS+=("$JID")
            done
        done
    done

    # Dependent aggregator: walks the scatter dir, builds per-field tables.
    if [ ${#SCATTER_JIDS[@]} -gt 0 ]; then
        DEP=$(IFS=:; echo "${SCATTER_JIDS[*]}")
        n_jobs=$((n_jobs + 1))
        sbatch \
            --job-name="sc_summary" \
            --output="${PWD}/logs_field/sc_summary_%j.out" \
            --error="${PWD}/logs_field/sc_summary_%j.err" \
            --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 \
            --cluster=smp --partition=smp \
            --mem=4G --time=0:10:00 \
            --dependency=afterany:${DEP} \
            --wrap="
mkdir -p ${PWD}/logs_field
source /ix/pgivi/moe32/envs/my_env/bin/activate
module load python/ondemand-jupyter-python3.11
cd ${PWD}
python aggregate_scatter_metrics.py $SCATTER_OUT
"
    fi
fi

# ---- PDF: one job per field ----
if [[ " $PLOT_TYPES " == *" pdf "* ]]; then
    for FIELD in $FIELDS; do
        n_jobs=$((n_jobs + 1))
        JOB_NAME="pdf_${FIELD}"
        sbatch \
            --job-name="$JOB_NAME" \
            --output="${PWD}/logs_field/${JOB_NAME}_%j.out" \
            --error="${PWD}/logs_field/${JOB_NAME}_%j.err" \
            --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 \
            --cluster=smp --partition=smp \
            --mem=64G --time=2:00:00 \
            --wrap="
mkdir -p ${PWD}/logs_field
source /ix/pgivi/moe32/envs/my_env/bin/activate
module load python/ondemand-jupyter-python3.11
cd ${PWD}
python plot_field_pdf.py \
    --field $FIELD \
    --timestep $TIMESTEP \
    --dns-data-dir $DNS_DATA_DIR \
    --mps-root $MPS_ROOT \
    --out-dir $PDF_OUT \
    --mode $MODE --fd-order $FD_ORDER --periodic-xz $PERIODIC_XZ \
    \$([ -d \"$LES_PLT\" ] && echo \"--les-plt $LES_PLT\")
"
    done
fi

# ---- CONTOUR: one job per field ----
if [[ " $PLOT_TYPES " == *" contour "* ]]; then
    for FIELD in $FIELDS; do
        n_jobs=$((n_jobs + 1))
        JOB_NAME="ct_${FIELD}"
        sbatch \
            --job-name="$JOB_NAME" \
            --output="${PWD}/logs_field/${JOB_NAME}_%j.out" \
            --error="${PWD}/logs_field/${JOB_NAME}_%j.err" \
            --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 \
            --cluster=smp --partition=smp \
            --mem=64G --time=2:00:00 \
            --wrap="
mkdir -p ${PWD}/logs_field
source /ix/pgivi/moe32/envs/my_env/bin/activate
module load python/ondemand-jupyter-python3.11
cd ${PWD}
python plot_field_contour.py \
    --field $FIELD \
    --timestep $TIMESTEP \
    --dns-data-dir $DNS_DATA_DIR \
    --mps-root $MPS_ROOT \
    --out-dir $CONTOUR_OUT \
    --mode $MODE --fd-order $FD_ORDER --periodic-xz $PERIODIC_XZ \
    \$([ -d \"$LES_PLT\" ] && echo \"--les-plt $LES_PLT\")
"
    done
fi

echo
echo "Total: $n_jobs jobs submitted."
echo
echo "Output directories:"
echo "  Scatter:  $SCATTER_OUT"
echo "  PDF:      $PDF_OUT"
echo "  Contour:  $CONTOUR_OUT"
echo
echo "Logs in: ${PWD}/logs_field/"
