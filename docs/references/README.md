# Numerical references

Keep reference source in its historical location rather than maintaining copies:

- MPS calculation and JHS MAT loader: `legacy/Julia/0mps_calc.jl`, functions
  `mps_truncate` and `load_velocity_1024`.
- Disjoint block filter: `src/les/block.jl`, shared with
  `legacy/LES_interface/LES_filter.jl`.
- Periodic centered filter: `src/les/periodic.jl`, shared with
  `legacy/Julia/LES_filter.jl`.

Public APIs are exported by `src/TurbulenceTools.jl`. Independent numerical
comparison oracles are pinned in `test/julia/` to the pre-refactor Git commit;
they do not follow changes to the working tree.
