#!/bin/bash
#SBATCH --job-name=SValsU            # Job name
#SBATCH --output=report_U_%j.out       # Standard output file
#SBATCH --nodes=1                   # Request one node
#SBATCH --ntasks-per-node=1         # Request one task per node (single task)
#SBATCH --cpus-per-task=20          # Request n  CPU cores
#SBATCH --cluster=smp               # Use SMP cluster
#SBATCH --partition=high-mem        # Partition with high memory
#SBATCH --mem=256G                  # Request 1024 GB of memory
#SBATCH --time=23:00:00             # Time limit: 23 hours

# Set the number of OpenMP threads to match the number of CPU cores requested
export OMP_NUM_THREADS=20

# Load Modules

module load julia/1.10.4
julia 1sv_U.jl

