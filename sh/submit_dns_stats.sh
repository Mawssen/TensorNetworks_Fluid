#!/bin/bash
#SBATCH --job-name=dns_stats
#SBATCH --output=dns_stats_%j.out
#SBATCH --error=dns_stats_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --mem=180G
#SBATCH --time=2:00:00

# -----------------------------------------------------------------------------
# DNS-only statistics for the PeleLM-FDF paper (timestep 0228, 512^3 cube).
# Loads 7+ fields of 512^3 float32 -> ~3.5 GB in memory at minimum.
# 64G of RAM gives comfortable headroom once optional fields are loaded.
# -----------------------------------------------------------------------------

# --- environment ---
source /ix/pgivi/moe32/envs/my_env/bin/activate
module load texlive/2021
module load python/ondemand-jupyter-python3.11

# --- paths (edit these) ---
# Data files for timestep <ts> live in $SLURM_SUBMIT_DIR/jet_<ts>/.
# Override DATA_DIR explicitly if your data lives elsewhere.
TIMESTEP="${TIMESTEP:-0198}"
DATA_DIR="${DATA_DIR:-$SLURM_SUBMIT_DIR/jet_${TIMESTEP}}"
OUT_DIR="${OUT_DIR:-$SLURM_SUBMIT_DIR/results/dns_${TIMESTEP}}"

mkdir -p "$OUT_DIR"
echo "DATA_DIR=$DATA_DIR"
echo "TIMESTEP=$TIMESTEP"
echo "OUT_DIR=$OUT_DIR"
which python

# --- analysis options ---
# REGION: 'cube' (fast, 512^3), 'full' (entire 864x1008x576), 'both' (default)
REGION="${REGION:-both}"
# FD_ORDER: spatial order for chi reconstruction (2, 4, 6, or 8). Default 8.
FD_ORDER="${FD_ORDER:-8}"
# PERIODIC_XZ: treat the cube as periodic in x and z (true|false). Default false
# -- the cube is a centered subset of the full periodic domain, so its
# boundaries are NOT true periodic neighbors. Set true only as a diagnostic.
PERIODIC_XZ="${PERIODIC_XZ:-false}"
echo "REGION=$REGION"
echo "FD_ORDER=$FD_ORDER"
echo "PERIODIC_XZ=$PERIODIC_XZ"

# --- compute statistics ---
python run_dns_stats.py "$DATA_DIR" "$TIMESTEP" "$OUT_DIR" \
    --region "$REGION" --fd-order "$FD_ORDER" \
    --periodic-xz "$PERIODIC_XZ" \
    --validate --sweep --validate-full
STATS_RC=$?

# --- generate figures ---
# plot_dns_stats.py accepts either a specific pkl or the parent dir
# containing cube/ and full/ subfolders. Passing the parent lets it
# figure out which region pickles exist and plot each into its own
# subdirectory.
if [ $STATS_RC -eq 0 ]; then
    python plot_dns_stats.py "$OUT_DIR" "$OUT_DIR/figures"
    PLOT_RC=$?
    if [ $PLOT_RC -ne 0 ]; then
        echo "PLOT STEP FAILED with rc=$PLOT_RC"
        echo "Stats are saved; you can re-run plotting manually:"
        echo "  python plot_dns_stats.py $OUT_DIR $OUT_DIR/figures"
        exit $PLOT_RC
    fi
else
    echo "Skipping plot step (stats rc=$STATS_RC)"
    exit $STATS_RC
fi

echo "Done."
