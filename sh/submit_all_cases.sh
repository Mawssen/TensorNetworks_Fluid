#!/bin/bash
# -----------------------------------------------------------------------------
# Submit MPS analysis cases for a single timestep across the full sweep:
#
# By default this submits one job per (mode, fd_order, periodic, sp).
# Each job is itself a 4-task array (one per cutoff: 1e-2, 1e-3, 1e-4, 1e-5)
# with an automatic comparison-plot follow-up at the end.
#
# Customize via env vars:
#   TIMESTEP        (default 0198)
#   MODES           (default "stored mixed recon")
#   FD_ORDERS       (default "8 2"; only used for mixed/recon)
#   PERIODICS       (default "false true"; only used for mixed/recon)
#   SPS             (default "1 2 3 4")
#
# Examples:
#   ./submit_all_cases.sh                          # everything
#   MODES="stored" ./submit_all_cases.sh            # stored only, all sps
#   SPS="1" ./submit_all_cases.sh                   # sp=1 only, all modes
#   MODES="recon" SPS="1 2" FD_ORDERS="8" ./submit_all_cases.sh
# -----------------------------------------------------------------------------

set -euo pipefail

TIMESTEP="${TIMESTEP:-0198}"
MODES="${MODES:-stored mixed recon}"
FD_ORDERS="${FD_ORDERS:-8 2}"
PERIODICS="${PERIODICS:-false true}"
SPS="${SPS:-1 2 3 4}"

echo "Submitting MPS cases for timestep $TIMESTEP"
echo "  MODES     = $MODES"
echo "  FD_ORDERS = $FD_ORDERS"
echo "  PERIODICS = $PERIODICS"
echo "  SPS       = $SPS"
echo

n=0
for SP in $SPS; do
    for MODE in $MODES; do
        if [ "$MODE" = "stored" ]; then
            n=$((n + 1))
            echo "[$n] stored  sp=$SP"
            Z_CHI_MODE=stored SP="$SP" TIMESTEP="$TIMESTEP" \
                sbatch submit_mps_stats.sh
        else
            for FD in $FD_ORDERS; do
                for PER in $PERIODICS; do
                    n=$((n + 1))
                    echo "[$n] $MODE  FD_ORDER=$FD  PERIODIC_XZ=$PER  sp=$SP"
                    Z_CHI_MODE="$MODE" FD_ORDER="$FD" PERIODIC_XZ="$PER" \
                        SP="$SP" TIMESTEP="$TIMESTEP" \
                        sbatch submit_mps_stats.sh
                done
            done
        fi
    done
done

echo
echo "Total: $n jobs submitted (each = 4-task array + comparison)."
echo
echo "After all stats finish, generate cross-sp overlays with:"
echo "  python plot_sp_overlay.py results results/sp_overlay_${TIMESTEP}_stored \\"
echo "      --mode stored --timestep $TIMESTEP"
echo "  python plot_sp_overlay.py results results/sp_overlay_${TIMESTEP}_recon_o8 \\"
echo "      --mode recon --fd-order 8 --timestep $TIMESTEP"
echo "  python plot_sp_overlay.py results results/sp_overlay_${TIMESTEP}_mixed_o8 \\"
echo "      --mode mixed --fd-order 8 --timestep $TIMESTEP"
