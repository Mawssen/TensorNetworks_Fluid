#!/bin/bash
#SBATCH --job-name=jhs-u-cf1e4-il
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --account=pgivi
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --chdir=/ix/pgivi/moe32/projects/Turbulence_TN_LES

set -euo pipefail
: "${TURBULENCE_SUBMISSION_DIRECTORY:?Set an absolute fresh directory for this submission}"
export TURBULENCE_RUN_DIRECTORY="$TURBULENCE_SUBMISSION_DIRECTORY/result"
export OMP_NUM_THREADS=10 JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
julia_bin=/ihome/pgivi/moe32/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia
hostname
printf 'SLURM_JOB_ID=%s\n' "$SLURM_JOB_ID"
date --iso-8601=seconds
"$julia_bin" --project=. --startup-file=no --compiled-modules=existing test/julia/compare_mps_cutoff.jl >"$TURBULENCE_SUBMISSION_DIRECTORY/cutoff_test.log" 2>&1
printf '%s\n' 'Cutoff regression passed; starting the approved full JHS case.'
"$julia_bin" --project=. --startup-file=no --compiled-modules=existing cases/mps/jhs_u_t1_cutoff1e4_interleaved.jl
date --iso-8601=seconds
