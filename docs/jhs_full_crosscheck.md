# Full-size JHS u optimization cross-check

After reviewing these results, the owner approved Float64 binned spectra and
chunked fidelity/L2 as preferred defaults in the new Python statistics API.
Original functions remain selectable with `implementation="reference"`.
The original CPU derivative remains default. Historical scripts and the
comparison harnesses continue to call their explicit original versions; no
general-purpose tolerance or Float32 default was introduced.

`cases/benchmarks/jhs_u_1024.py` compares the preserved and optimized kernels on
the original U_t1/U1 field and its completed 1024³ cutoff-1e-4 reconstruction.
The existing Python JHS loader convention, transpose(2,1,0), is applied to both.
No crop, mean subtraction or precision conversion beyond the historical
Float64 load is added at the kernel boundary. This validates kernels, not the
complete mean-centered three-component 2FI scalar workflow.

Scope:

- First-order periodic derivatives on all axes, on DNS and reconstructed u,
  using the literal historical 2FI spacing 0.0061359406.
- Historical versus chunked squared-overlap fidelity and relative L2 for the
  original/reconstructed pair.
- With `--spectrum dns-uvw`, use the actual full-resolution `U1`, `V1` and `W1`
  DNS fields for the three-component spectrum comparison. This is not a mixed
  DNS/MPS field and does not require new v/w reconstructions.
- If explicitly approved with `--spectrum u-only`, compare the historical
  Float64, new Float64 and historical Float32 spectrum implementations on
  `(u,0,0)` for each field. These are u-component contributions, not total
  three-component spectra. `--spectrum defer` omits this step instead.

Separate worker processes measure each version's elapsed calculation time and
Linux peak RSS; loading and output times are recorded separately. One full-size
call per version avoids unapproved repeated production calculations. Timings
are single observations, not warmed medians. Temporary derivative arrays permit
complete elementwise comparison and are removed afterward. Scalar/spectrum
outputs, differences, source snapshots/hashes, parameter records and worker logs
are retained under a new ignored artifact directory. No tolerance is selected.

Print the plan without loading data:

```sh
python3 -B cases/benchmarks/jhs_u_1024.py --plan \
  --output-directory artifacts/benchmarks/jhs-u-1024/NEW_RUN \
  --spectrum dns-uvw
```

The Slurm template proposes smp/smp, pgivi, short QoS, one CPU, 256 GiB and
three hours. Its Python module has been checked to import NumPy and h5py without
installation. All old/new versions use the same environment and thread count.
The memory/time request is not a measured performance estimate for these full
statistics; historical shell scans and full FFTs can be expensive.

Submission and the spectrum choice require owner approval. This does not
compress v/w, change original files or replace production defaults.

## Completed full-size results — 2026-11-16

Job `24749966` completed with exit `0:0` on `smp-n219` in 01:00:24, using one
CPU and the existing Python module (NumPy 2.4.6, h5py 3.16.0). Slurm recorded
104.2 GiB batch peak RSS against the approved 256 GiB limit. All eight tasks
completed and all six source hashes remained unchanged. Timings below are one
calculation per version, excluding load and temporary-file output time; these
are observed case results, not repeated benchmark medians.

| Kernel / input | Historical calculation | New calculation | Worker peak RSS, old → new | Cross-check |
| --- | ---: | ---: | ---: | --- |
| Three-component DNS spectrum | 1153.92 s | 174.83 s (6.60× faster) | 88.6 → 56.6 GiB | max relative difference 2.64e-15 |
| DNS-u/MPS-u fidelity and L2 | 107.38 s | 34.47 s (3.11× faster) | 48.6 → 16.6 GiB | max relative difference 1.55e-12 |
| Derivatives, all axes of DNS/MPS u | 13.66–28.63 s | 19.66–92.68 s | about 56 → 16–17 GiB | all six arrays bitwise identical |

The slabbed derivatives were slower in every full-size calculation. Their role
is currently an opt-in memory-saving alternative, not an established CPU speed
optimization for these fields. See the per-axis table in `staged_statistics.md`.

The existing Float32 spectrum variant took 158.57 s with 64.6 GiB worker peak
RSS, and differed from the historical Float64 spectrum by at most 9.73e-9
relative (2.04e-9 absolute). The new Float64 spectrum differed by 2.64e-15
relative (2.78e-17 absolute). Wavenumbers matched exactly for all three versions;
all outputs were finite with no zero-reference mismatches. No approximate
scientific acceptance tolerance was selected and no defaults were replaced.

The reference raw-u comparison returned squared-overlap fidelity
0.9982175306751173 and relative L2 0.04221930036336234. The chunked result was
0.9982175306766636 and 0.042219300363398084, respectively. These compare the
original/reconstructed u arrays without the external mean subtraction used by
the complete historical 2FI workflow.

The spectrum used actual DNS U1/V1/W1 fields. No MPS v/w were generated, and
no mixed DNS/MPS three-component result was constructed. This is not a full
three-component reconstructed-flow scalar-statistics run.

Saved artifacts:
`artifacts/benchmarks/jhs-1024-uvw/submission-20p02r/` contains scheduler/worker
logs, source snapshots, input metadata, all scalar/spectrum outputs and the
complete `comparison/report.json`. Temporary gradient fields were removed only
after every element had been compared; original data/reconstructions are intact.
