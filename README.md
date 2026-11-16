# TurbulenceTools

From the repository root, use the checked-in Julia project and lockfile:

```bash
cd /ix/pgivi/moe32/projects/Turbulence_TN_LES
JULIA_PKG_PRECOMPILE_AUTO=0 julia --project=. --startup-file=no --compiled-modules=existing -e 'using TurbulenceTools; println("TurbulenceTools loaded")'
```

Inside an **existing CRC compute allocation**, run the two cases to copy:

```bash
julia --project=. examples/mps_case.jl
julia --project=. examples/les_case.jl
```

These use tiny synthetic inputs and create new `constant-mps.mat` and
`les-case.mat` files. Existing output paths are refused. Edit the grouped values
at the top of each case for a new dataset after its scientific inputs are
settled. No command here allocates resources or submits a job.

The lockfile reuses the dependency versions already recorded in the existing
CRC Julia v1.12 environment, retaining only the package's required dependency
closure. No dependencies were installed or upgraded to create it. On a fresh
machine, after dependency installation is approved, setup is:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Use Julia 1.12.6 to match the existing CRC environment. The package retains the
historical MKL backend and BLAS thread count of 10 at import. Package
precompilation is disabled so those historical process settings are applied on
each import; dependency caches can still be used. CRC resource settings remain
outside the package.

## Public API and case scripts

`src/TurbulenceTools.jl` is the single package entry point:

```julia
using TurbulenceTools

field = load_centered_be_float32(input_path; full_shape=shape, crop_size=crop)
reconstructed, bonds = compress_reconstruct(field; chi=chi, ordering=ordering)
write_mps_mat(output_path, reconstructed, bonds)

coarse = block_average(field64; block_size=delta)
smoothed = periodic_box_smooth(field; radii=(rx, ry, rz))
```

The package loads the shared implementations in `Julia/TurbulenceMPS.jl`,
`Julia/TurbulenceLES.jl`, and the historical numerical sources. Existing Julia
entry points remain available; no numerical algorithm is duplicated for the
package. Public docstrings describe shapes, axes, parameters and return values.

Copy [examples/mps_case.jl](examples/mps_case.jl) for MPS. It accepts an in-memory
field from your loader, or uses the existing big-endian Float32 binary loader.
Input path, full-grid shape, crop, chi, ordering, and output are case values.
The loader preserves x-fastest storage and centered cropping; the CLI's Jet
Flame defaults remain `(864,1008,576)` and a `512³` crop. Compression performs
no additional normalization.

Copy [examples/les_case.jl](examples/les_case.jl) for LES. Select `:block` or
`:periodic` explicitly near the top. Its optional MAT loader uses an explicitly
named variable without changing dtype or axes. Output writing, spacing metadata
and dataset settings stay in the case. Establish dataset axis mapping, physical
units and coordinates before using production data.

- `block_average` uses disjoint blocks starting at index 1, with no wrapping or
  partial edge blocks. It requires cubic Float64 input with side N divisible by
  positive block size δ and returns `(N÷δ)³`. Uniform spacing becomes δ times the
  input spacing. Each result represents a block average; a centroid coordinate
  starts `(δ-1)/2` input spacings beyond the first input coordinate.
- `periodic_box_smooth` uses the direct historical centered stencil with widths
  `(2rx+1,2ry+1,2rz+1)`, wrapping every axis periodically. It retains shape,
  element type, coordinates and spacing. Radii are nonnegative integers.
  The distinct separable/sampling variant remains in the historical script.

Both preserve input axis order. Neither infers physical axes or units.

## Other existing invocations

All Julia example and validation commands use `--project=.`:

```bash
julia --project=. examples/mps_case.jl pattern pattern-mps.mat
julia --project=. examples/les_synthetic.jl block
julia --project=. examples/les_synthetic.jl periodic
```

The CLI-driven MAT LES case is also retained:

```bash
julia --project=. examples/les_mat_case.jl INPUT.mat VARIABLE block DELTA DX DY DZ NEW_OUTPUT.mat
julia --project=. examples/les_mat_case.jl INPUT.mat VARIABLE periodic RX,RY,RZ DX DY DZ NEW_OUTPUT.mat
```

Replace uppercase tokens with approved case values; these are templates, not
production selections. The fixed-layout historical MPS runner and dry run remain:

```bash
python3 -B scripts/mps_v1_dry_run.py /absolute/path/to/case.json
python3 -B scripts/mps_v1_run.py /absolute/path/to/approved-case.json
```

See [runner usage](docs/mps_v1_execution.md) for cwd, logs and output naming.

## Verification

Lightweight Python checks:

```bash
python3 -B -m unittest discover -s tests -p 'test_mps_v1*.py' -v
```

Numerical checks, from the repository root inside an existing compute allocation:

```bash
julia --project=. validation/compare_mps_api.jl
julia --project=. validation/compare_les_api.jl
```

The comparisons use the pinned pre-refactor commit
`28149867b368f5b576fc41e82b4ba1425d38a21d`, so committing the package does not
change their oracle. Keep that commit in the clone's history. MPS changes only
test dimensions in the oracle's source in memory, then checks axes/crop,
orderings 1–4, chi 1/3, reconstruction and MAT output. LES checks expected block
means, coarse shape, periodic boundary/corner wraparound, and historical kernels.
Comparisons use exact equality; no approximate tolerance is invented.
