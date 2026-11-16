#!/bin/bash
# Proposed full-size comparison limits; owner approval required for submission.
#SBATCH --job-name=jhs-u-crosscheck
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --account=pgivi
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --chdir=/ix/pgivi/moe32/projects/Turbulence_TN_LES
set -euo pipefail
: "${TURBULENCE_SUBMISSION_DIRECTORY:?Supply a fresh directory}"
: "${TURBULENCE_SPECTRUM_SCOPE:?Explicitly choose dns-uvw, u-only or defer}"
module load python/ondemand-jupyter-python3.11
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
hostname
date --iso-8601=seconds
python -B cases/benchmarks/jhs_u_1024.py \
    --output-directory "$TURBULENCE_SUBMISSION_DIRECTORY/comparison" \
    --spectrum "$TURBULENCE_SPECTRUM_SCOPE"
date --iso-8601=seconds
