#!/bin/bash
#SBATCH --job-name=qc4pde_metrics
#SBATCH --output=qc4pde_metrics_%j.out
#SBATCH --error=qc4pde_metrics_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --mem=32G
#SBATCH --time=0:30:00

# -----------------------------------------------------------------------------
# Compute QC4PDE Z-field metrics (infidelity + mean/RMS profile errors).
# Output: QC4PDE/metrics_Z.txt
# -----------------------------------------------------------------------------

source /ix/pgivi/moe32/envs/my_env/bin/activate
module load python/ondemand-jupyter-python3.11

cd "$SLURM_SUBMIT_DIR"

TIMESTEP="${TIMESTEP:-0198}"
SP="${SP:-1}"
FD_ORDER="${FD_ORDER:-4}"
PERIODIC_XZ="${PERIODIC_XZ:-false}"
RESULTS_ROOT="${RESULTS_ROOT:-$SLURM_SUBMIT_DIR/results}"
OUT_DIR="${OUT_DIR:-$SLURM_SUBMIT_DIR/QC4PDE}"
DNS_DATA_DIR="${DNS_DATA_DIR:-$SLURM_SUBMIT_DIR/jet_${TIMESTEP}}"
MPS_ROOT="${MPS_ROOT:-$SLURM_SUBMIT_DIR}"
PEPS_DIR="${PEPS_DIR:-$SLURM_SUBMIT_DIR/truncated_${TIMESTEP}/PEPS}"

echo "TIMESTEP     = $TIMESTEP"
echo "SP           = $SP"
echo "FD_ORDER     = $FD_ORDER"
echo "PERIODIC_XZ  = $PERIODIC_XZ"
echo "RESULTS_ROOT = $RESULTS_ROOT"
echo "OUT_DIR      = $OUT_DIR"
echo "DNS_DATA_DIR = $DNS_DATA_DIR"
echo "MPS_ROOT     = $MPS_ROOT"
echo "PEPS_DIR     = $PEPS_DIR"

python compute_qc4pde_metrics.py "$RESULTS_ROOT" "$OUT_DIR" \
    --timestep "$TIMESTEP" \
    --dns-data-dir "$DNS_DATA_DIR" \
    --peps-dir "$PEPS_DIR" \
    --mps-root "$MPS_ROOT" \
    --sp "$SP" \
    --fd-order "$FD_ORDER" \
    --periodic-xz "$PERIODIC_XZ"
RC=$?
echo "Done with rc=$RC"
exit $RC
