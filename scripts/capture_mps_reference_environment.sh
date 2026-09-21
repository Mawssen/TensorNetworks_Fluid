#!/usr/bin/env bash
# Read-only environment capture for the provisional MPS reference.
# This script reports state. It does not load modules, install/update packages,
# execute repository Julia sources, submit jobs, or change the environment.

set -u

REFERENCE_JULIA="${REFERENCE_JULIA:-${HOME}/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia}"

printf '%s\n' '== Capture context =='
date --iso-8601=seconds
hostname
uname -a
printf 'working_directory=%s\n' "$(pwd)"
printf 'reference_julia=%s\n' "$REFERENCE_JULIA"

printf '%s\n' '== Module environment =='
if type module >/dev/null 2>&1; then
    module list 2>&1
else
    printf '%s\n' 'module command unavailable'
fi
printf 'LOADEDMODULES=%s\n' "${LOADEDMODULES-<unset>}"
printf 'MODULEPATH=%s\n' "${MODULEPATH-<unset>}"

printf '%s\n' '== Julia, package, BLAS, and thread environment =='
if [[ -x "$REFERENCE_JULIA" ]]; then
    "$REFERENCE_JULIA" --startup-file=no --history-file=no -e '
        using InteractiveUtils
        using LinearAlgebra
        using Pkg

        println("VERSION=", VERSION)
        println("Sys.BINDIR=", Sys.BINDIR)
        println("Base.active_project()=", Base.active_project())
        println("LOAD_PATH=", LOAD_PATH)
        versioninfo(verbose=true)
        println("Pkg.status(manifest mode):")
        Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)
        println("BLAS.vendor()=", BLAS.vendor())
        println("BLAS.get_config()=", BLAS.get_config())
        println("BLAS.get_num_threads()=", BLAS.get_num_threads())
        println("Threads.nthreads()=", Threads.nthreads())
        println("Threads.nthreadpools()=", Threads.nthreadpools())
        println("Threads.threadpoolsize(:default)=", Threads.threadpoolsize(:default))
    '
else
    printf '%s\n' 'reference Julia executable is not available or not executable'
fi
printf 'JULIA_PROJECT=%s\n' "${JULIA_PROJECT-<unset>}"
printf 'JULIA_LOAD_PATH=%s\n' "${JULIA_LOAD_PATH-<unset>}"
printf 'JULIA_DEPOT_PATH=%s\n' "${JULIA_DEPOT_PATH-<unset>}"
printf 'JULIA_NUM_THREADS=%s\n' "${JULIA_NUM_THREADS-<unset>}"
printf 'OMP_NUM_THREADS=%s\n' "${OMP_NUM_THREADS-<unset>}"
printf 'MKL_NUM_THREADS=%s\n' "${MKL_NUM_THREADS-<unset>}"
printf 'OPENBLAS_NUM_THREADS=%s\n' "${OPENBLAS_NUM_THREADS-<unset>}"

printf '%s\n' '== Slurm environment =='
env | LC_ALL=C sort | sed -n '/^SLURM_/p'
if [[ -n "${SLURM_JOB_ID-}" ]] && command -v scontrol >/dev/null 2>&1; then
    scontrol show job --details "$SLURM_JOB_ID"
else
    printf '%s\n' 'No active Slurm job ID available for scontrol inspection'
fi
