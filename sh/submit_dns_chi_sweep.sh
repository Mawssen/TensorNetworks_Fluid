#!/bin/bash
#SBATCH --job-name=dns_chi_sweep
#SBATCH --output=dns_chi_sweep_%j.out
#SBATCH --error=dns_chi_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --mem=64G
#SBATCH --time=5:00:00

# -----------------------------------------------------------------------------
# Sweep DNS chi reconstruction across (order, periodic_xz) combinations
# on the 512^3 cube. Loads cube fields once, runs 6 reconstructions
# (order in {2,4,8} x periodic in {false,true}), prints a table, and
# saves it.
#
# Submit examples:
#   sbatch submit_dns_chi_sweep.sh                 # default ts=0198
#   TIMESTEP=0099 sbatch submit_dns_chi_sweep.sh
# -----------------------------------------------------------------------------

source /ix/pgivi/moe32/envs/my_env/bin/activate
module load texlive/2021
module load python/ondemand-jupyter-python3.11

TIMESTEP="${TIMESTEP:-0198}"
DATA_DIR="${DATA_DIR:-$SLURM_SUBMIT_DIR/jet_${TIMESTEP}}"
OUT_DIR="${OUT_DIR:-$SLURM_SUBMIT_DIR/results/dns_${TIMESTEP}/chi_sweep}"

echo "TIMESTEP = $TIMESTEP"
echo "DATA_DIR = $DATA_DIR"
echo "OUT_DIR  = $OUT_DIR"
which python

mkdir -p "$OUT_DIR"

python sweep_dns_chi_recon.py "$DATA_DIR" "$TIMESTEP" "$OUT_DIR"
RC=$?
if [ $RC -ne 0 ]; then
    echo "FAILED rc=$RC"
    exit $RC
fi
echo "Done."
