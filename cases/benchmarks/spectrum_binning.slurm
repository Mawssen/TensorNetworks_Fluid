#!/bin/bash
#SBATCH --job-name=spectrum-binning
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --account=pgivi
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --chdir=/ix/pgivi/moe32/projects/Turbulence_TN_LES

set -euo pipefail
: "${TURBULENCE_SUBMISSION_DIRECTORY:?Supply a fresh submission directory}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
hostname
date --iso-8601=seconds
/usr/bin/python3 -B cases/benchmarks/spectrum_binning.py \
    --output-directory "$TURBULENCE_SUBMISSION_DIRECTORY/comparison" \
    --sizes 32 64 128 --repetitions 3
date --iso-8601=seconds
