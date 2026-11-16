# Historical workflows

This directory preserves the earlier research tree, grouped in one place:

- `Julia/`: historical MPS and analysis entry points. The package still loads
  `Julia/0mps_calc.jl` as its numerical source through `src/mps/api.jl`.
- `LES_interface/`: earlier LES interfaces and launch helpers.
- `mps_py_codes/`: original Python numerical routines.
- `py/`: Python analysis and plotting scripts.
- `sh/`, `Slurms/`: earlier shell and Slurm launchers.
- `notebooks/`: exploratory notebooks.

Relocation does not certify the old dataset paths or launch environments. These
scripts retain their original numerical content and dataset-relative paths;
standalone workflows may need their original working directory or explicitly
approved data paths. Use `cases/` and the package for new work. The two LES
historical files include the same shared kernels under `src/les/`.
