#!/bin/bash
#SBATCH --job-name=mps_cr_match
#SBATCH --output=report_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --array=0-27%10

# -----------------------------------------------------------------------------
# Sweep three cutoffs and one bond-dimension cap to bracket LES CR=512.
# All for mixfrac, sp=1 (interleaved). 4 total tasks.
# -----------------------------------------------------------------------------

export OMP_NUM_THREADS=10
JULIA=$HOME/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia

#FILENAMES=("jet_mixfrac_0198.dat")
FILENAMES=("jet_chi_0198.dat" "jet_Y_H2_0198.dat" 
"jet_Y_CO2_0198.dat" "jet_Y_HCO_0198.dat" "jet_Y_HO2_0198.dat" 
"jet_Y_H_0198.dat" "jet_Y_O2_0198.dat" "jet_Y_OH_0198.dat" 
"jet_Y_O_0198.dat" "jet_alpha_0198.dat" "jet_rho_0198.dat"
"jet_Y_H2O_0198.dat" "jet_Y_CO_0198.dat" "jet_T_0198.dat")


# MODES: each entry is either a cutoff (no prefix) or "chi=<N>" for a max-bond-dim run.
MODES=("4.1e-4" "chi=93")
SPLITS=("1")

NF=${#FILENAMES[@]}
NM=${#MODES[@]}
NS=${#SPLITS[@]}

t=$SLURM_ARRAY_TASK_ID
sp_idx=$(( t % NS ))
md_idx=$(( (t / NS) % NM ))
fn_idx=$(( t / (NS * NM) ))

FILENAME=${FILENAMES[$fn_idx]}
MODE=${MODES[$md_idx]}
SP=${SPLITS[$sp_idx]}

STEM="${FILENAME%.dat}"          # jet_mixfrac_0198
TIMESTEP="${STEM##*_}"           # 0198
INPUT_PATH="./jet_${TIMESTEP}/${FILENAME}"

echo "Task $t: input=$INPUT_PATH, mode=$MODE, sp=$SP"
$JULIA 0mps_calc.jl "$INPUT_PATH" "$MODE" "$SP"
