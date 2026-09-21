#!/bin/bash
#SBATCH --job-name=stld_e2
#SBATCH --output=report_pdf_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=smp
#SBATCH --partition=high-mem
#SBATCH --mem=512G
#SBATCH --time=23:00:00


# Load Modules

#module load gcc/8.2.0
#module load python/anaconda3.10-2022.10
source /ix/pgivi/moe32/envs/my_env/bin/activate
module load texlive/2021
module load python/ondemand-jupyter-python3.11
python 6FI_MPS_3D_1024cubed_t1_stilde.py

