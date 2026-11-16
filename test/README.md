# Tests

- `test/python/test_mps_v1*.py`: lightweight planner/runner tests with no scientific data.
- `test/julia/compare_mps_api.jl`: historical chi comparisons on tiny fields.
- `test/julia/compare_mps_cutoff.jl`: historical cutoff-only comparison at 1e-4.
- `test/julia/compare_les_api.jl`: block averages and periodic boundary checks.
- `test/python/test_spectrum_binning.py`: exact shell/count and synthetic checks
  for the opt-in Float64 spectrum optimization. Numerical cases require compute.

From the repository root:

```sh
python3 -B -m unittest discover -s test/python -p 'test_mps_v1*.py'
```

Julia numerical checks require a compute allocation, using `julia --project=.`.
`julia --project=. test/runtests.jl` runs all three existing Julia comparisons.
Generated run outputs and logs belong under ignored `artifacts/runs/`, not alongside tests.
