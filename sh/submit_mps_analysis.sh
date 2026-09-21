#!/bin/bash
#SBATCH --job-name=mps_analysis
#SBATCH --output=mps_analysis_%j.out
#SBATCH --error=mps_analysis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --mem=64G
#SBATCH --time=2:00:00

# -----------------------------------------------------------------------------
# Aggregate CR / fidelity / L2 / max(chi_crit) for every truncated field
# at every cutoff into a single Excel file with one sheet per cutoff.
#
# Submit examples:
#   sbatch submit_mps_analysis.sh                    # default ts=0198
#   TIMESTEP=0099 sbatch submit_mps_analysis.sh
# -----------------------------------------------------------------------------

source /ix/pgivi/moe32/envs/my_env/bin/activate
module load texlive/2021
module load python/ondemand-jupyter-python3.11

TIMESTEP="${TIMESTEP:-0198}"
SP="${SP:-1}"

# Locate input dir. Layout: truncated_<TS>/sp<SP>/.  Legacy fallbacks.
DEFAULT_TRUNC_DIR="$SLURM_SUBMIT_DIR/truncated_${TIMESTEP}/sp${SP}"
if [ ! -d "$DEFAULT_TRUNC_DIR" ]; then
    LEGACY_FLAT="$SLURM_SUBMIT_DIR/truncated_${TIMESTEP}_sp${SP}"
    LEGACY_NOSP="$SLURM_SUBMIT_DIR/truncated_${TIMESTEP}"
    if [ -d "$LEGACY_FLAT" ]; then
        DEFAULT_TRUNC_DIR="$LEGACY_FLAT"
    elif [ "$SP" = "1" ] && [ -d "$LEGACY_NOSP" ]; then
        DEFAULT_TRUNC_DIR="$LEGACY_NOSP"
    fi
fi
TRUNCATED_DIR="${TRUNCATED_DIR:-$DEFAULT_TRUNC_DIR}"
DNS_DIR="${DNS_DIR:-$SLURM_SUBMIT_DIR/jet_${TIMESTEP}}"
OUT_XLSX="${OUT_XLSX:-$SLURM_SUBMIT_DIR/results/mps_summary_${TIMESTEP}_sp${SP}.xlsx}"

echo "TIMESTEP      = $TIMESTEP"
echo "SP            = $SP"
echo "TRUNCATED_DIR = $TRUNCATED_DIR"
echo "DNS_DIR       = $DNS_DIR"
echo "OUT_XLSX      = $OUT_XLSX"
which python

mkdir -p "$(dirname "$OUT_XLSX")"

python analyze_mps_truncations.py \
    "$TRUNCATED_DIR" "$DNS_DIR" "$TIMESTEP" "$OUT_XLSX"
RC=$?
if [ $RC -ne 0 ]; then
    echo "FAILED rc=$RC"
    exit $RC
fi
echo "Done."
