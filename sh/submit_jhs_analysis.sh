#!/bin/bash
#SBATCH --job-name=jhs_analysis
#SBATCH --cluster=smp
#SBATCH --partition=high-mem
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=6:00:00
#SBATCH --output=jhs_analysis_%j.out

# JHS 1024^3 analysis, ONE case per submission (as requested).
#   case A: chi=27, les factor 32 -> LES 32^3
#   case B: chi=84, les factor 16 -> LES 64^3
# Edit CHI / FACTOR below, submit, then edit and submit again for the other.
#
# MEMORY: a 1024^3 float64 cube is 8.6 GB; three DNS + three MPS + the
# spectrum temporaries push this well past 100 GB, hence high-mem/480G.
# If it OOMs, add --skip-spectrum to get stats+contour first, then run the
# spectrum separately.

set -euo pipefail

module load python/ondemand-jupyter-python3.11

# CHI and FACTOR come from the environment so one script serves both cases:
#   sbatch --export=ALL,CHI=27,FACTOR=32 submit_jhs_analysis.sh
#   sbatch --export=ALL,CHI=84,FACTOR=16 submit_jhs_analysis.sh
CHI=${CHI:-27}
FACTOR=${FACTOR:-32}
echo "=== JHS analysis: chi=${CHI}, les factor=${FACTOR} ==="

JHSDIR=/ix/pgivi/moe32/Schmidt/Forced_Isotropic/5_Grid_1024/Forced_Isotropic_1024Cubed_mat_in_one
# save_to_mat writes the reconstruction into the directory the MPS job ran
# in (the generator's --rundir), so that is where the truncated_*.mat live.
MPSDIR=/ix/pgivi/moe32/Schmidt/Forced_Isotropic/5_Grid_1024/Codes/calculations/1Generate_truncated_velocity/1U_e2/cf_sweep
CACHE=/ix/pgivi/moe32/Schmidt/Forced_Isotropic/5_Grid_1024/jhs_filtered
OUT=QC4PDE_JHS

python jhs_velocity_spectrum_stats.py \
    --dns-dir "$JHSDIR" \
    --dns-pattern "{comp}_t1.mat" \
    --mps-dir "$MPSDIR" \
    --mps-pattern "truncated_{comp}_t1_1024_il_chi${CHI}.mat" \
    --mps-chi ${CHI} \
    --les-factor ${FACTOR} \
    --components U,V,W \
    --cache-dir "$CACHE" \
    --out-dir "$OUT"

echo "JHS case chi=${CHI} factor=${FACTOR} done. Outputs in ${OUT}/"
