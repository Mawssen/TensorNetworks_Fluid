#!/bin/bash
#SBATCH --job-name=PEPS_eq2           # Job name
#SBATCH --output=report_%j.out         # Standard output file
#SBATCH --nodes=1                   # Request one node
#SBATCH --ntasks-per-node=1         # Request one task per node (single task)
#SBATCH --cpus-per-task=20          # Request n  CPU cores
#SBATCH --cluster=smp               # Use SMP cluster
#SBATCH --partition=smp             # Partition with high memory
#SBATCH --mem=256G                  # Request 1024 GB of memory
#SBATCH --time=23:00:00             # Time limit: 23 hours

# Set the number of OpenMP threads to match the number of CPU cores requested
export OMP_NUM_THREADS=20

# Load Modules

source /ihome/pgivi/moe32/.bashrc
julia mps_calc.jl

