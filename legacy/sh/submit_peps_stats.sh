#!/bin/bash
#SBATCH --job-name=peps_stats
#SBATCH --output=peps_stats_%j.out
#SBATCH --error=peps_stats_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --mem=64G
#SBATCH --time=1:00:00

# -----------------------------------------------------------------------------
# One-shot SLURM job: compute PEPS stats. Reads
#   /ix/pgivi/moe32/Aidyn_DNS/stats/truncated_0198/PEPS/<var>_wf_D=9_periodic.mat
# for all 7 fields, computes stats, saves pickle to results/peps_0198/.
# -----------------------------------------------------------------------------

source /ix/pgivi/moe32/envs/my_env/bin/activate
module load texlive/2021
module load python/ondemand-jupyter-python3.11

TIMESTEP="${TIMESTEP:-0198}"
TAG="${TAG:-wf_D=9_periodic}"
Z_CHI_MODE="${Z_CHI_MODE:-mixed}"
FD_ORDER="${FD_ORDER:-4}"
PERIODIC_XZ="${PERIODIC_XZ:-false}"

DEFAULT_DATA_DIR="$SLURM_SUBMIT_DIR/truncated_${TIMESTEP}/PEPS"
DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
DNS_DATA_DIR="${DNS_DATA_DIR:-$SLURM_SUBMIT_DIR/jet_${TIMESTEP}}"

OUT_DIR="${OUT_DIR:-$SLURM_SUBMIT_DIR/results/peps_${TIMESTEP}}"
mkdir -p "$OUT_DIR"

echo "TIMESTEP     = $TIMESTEP"
echo "TAG          = $TAG"
echo "DATA_DIR     = $DATA_DIR"
echo "DNS_DATA_DIR = $DNS_DATA_DIR"
echo "Z_CHI_MODE   = $Z_CHI_MODE"
echo "FD_ORDER     = $FD_ORDER"
echo "PERIODIC_XZ  = $PERIODIC_XZ"
echo "OUT_DIR      = $OUT_DIR"

python run_peps_stats.py "$DATA_DIR" "$TIMESTEP" "$OUT_DIR" \
    --tag "$TAG" \
    --z-chi-mode "$Z_CHI_MODE" \
    --fd-order "$FD_ORDER" \
    --periodic-xz "$PERIODIC_XZ" \
    --dns-data-dir "$DNS_DATA_DIR"
RC=$?
echo "Done with rc=$RC"
exit $RC
