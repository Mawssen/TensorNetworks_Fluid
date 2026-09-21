#!/bin/bash
#SBATCH --job-name=chi_decomp
#SBATCH --output=chi_decomp_%j.out
#SBATCH --error=chi_decomp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --mem=64G
#SBATCH --time=1:00:00

# -----------------------------------------------------------------------------
# One-off SLURM job: DNS vs MPS chi=93 vs MPS 4.1e-4 decomposition of the
# scalar-dissipation-rate ingredients (Z, |grad Z|, |grad Z|^2, alpha).
#
# Produces profiles + PDFs + scatter plots + metric summary text.
# -----------------------------------------------------------------------------

source /ix/pgivi/moe32/envs/my_env/bin/activate
module load texlive/2021
module load python/ondemand-jupyter-python3.11

TIMESTEP="${TIMESTEP:-0198}"
SP="${SP:-1}"
FD_ORDER="${FD_ORDER:-4}"
PERIODIC_XZ="${PERIODIC_XZ:-false}"
OUT_DIR="${OUT_DIR:-$SLURM_SUBMIT_DIR/results/chi_decomp_${TIMESTEP}}"
DNS_DATA_DIR="${DNS_DATA_DIR:-$SLURM_SUBMIT_DIR/jet_${TIMESTEP}}"
MPS_ROOT="${MPS_ROOT:-$SLURM_SUBMIT_DIR}"

mkdir -p "$OUT_DIR"

python plot_chi_decomposition.py "$SLURM_SUBMIT_DIR/results" "$OUT_DIR" \
    --timestep "$TIMESTEP" \
    --dns-data-dir "$DNS_DATA_DIR" \
    --mps-root "$MPS_ROOT" \
    --sp "$SP" \
    --fd-order "$FD_ORDER" \
    --periodic-xz "$PERIODIC_XZ" \
    --cases PEPS,chi93,4.1e-4

echo "Done."
