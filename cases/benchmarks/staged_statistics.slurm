#!/bin/bash
#SBATCH --job-name=staged-statistics
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
: "${TURBULENCE_SUBMISSION_DIRECTORY:?Supply a fresh directory}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/usr/bin/python3 -B cases/benchmarks/staged_statistics.py --output-directory "$TURBULENCE_SUBMISSION_DIRECTORY/comparison"
