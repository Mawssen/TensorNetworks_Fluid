# Bounded-workspace statistics alternatives

Owner decision after the full-size cross-check: `compare_fields` now defaults
to the chunked implementation; `implementation="reference"` selects the old
functions. `spectral_derivative` retains the original CPU default, with slabbed
behavior available explicitly. This supersedes earlier opt-in/default-status
notes below; the recorded benchmark measurements remain unchanged.

New opt-in functions live in `src/python/turbulence_tools_stats/reductions.py`:

- `spectral_derivative_slabbed(field, axis, dx, order=1, slab_size=8)` keeps
  each complete FFT line and partitions another axis. It returns a full Float64
  array; the FFT workspace is slab-sized. Periodic boundaries, frequency order,
  spacing and derivative formula match the historical spectral derivative.
- `compare_fields_chunked(reference, candidate, slab_size=8)` returns the
  historical squared-overlap `fidelity` and `relative_l2` without full-field
  cast copies or a full-field difference array. Fields must already have the
  same shape, axes and Float64 dtype. No centering is inferred.

Both options preserve mathematical formulas but may change floating reduction
order. No acceptance tolerance or production default has been selected. Original
functions remain untouched in `legacy/mps_py_codes/`. The existing Float64
spectrum alternative is also retained.

Inside an approved compute allocation, from the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -B cases/benchmarks/staged_statistics.py \
  --output-directory artifacts/benchmarks/staged-statistics/NEW_RUN
```

The directory must be new. This performs synthetic cross-checks and separate
process benchmarks of the original and staged implementations at 128³, including
peak process RSS, median timing, differences, source snapshots and hashes.
`cases/benchmarks/staged_statistics.slurm` prepares a 1-CPU, 4-GiB, 10-minute
CRC job, requiring approval before submission. No production input is accepted.

GPU FFT/MPS implementations are not included: those require an approved GPU
environment, hardware allocation and dependency changes. Skipping gradient-file
outputs or replacing full three-component statistics would also require a
separate decision. These alternatives do neither.

## CRC cross-check — 2026-11-16

Job `24749910` completed with exit `0:0` on `smp-n219`. Inputs were synthetic
Float64 fields; timings are medians of three warmed calls in separate workers.

| Operation, 128³ | Original | Staged | Speedup | Worker peak RSS, original → staged |
| --- | ---: | ---: | ---: | ---: |
| Periodic derivative, axis 1 | 34.411 ms | 28.392 ms | 1.21× | 194.9 → 103.8 MiB |
| Fidelity and relative L2 together | 6.962 ms | 2.663 ms | 2.61× | 98.1 → 67.9 MiB |

All 36 derivative comparisons (three axes, orders 1/2, slab sizes 1/3/20,
odd/even non-cubic shapes and noncontiguous inputs) were bitwise identical.
The timed 128³ derivative output was also bitwise identical. Constant-field
derivatives, identical-field metrics, and zero-denominator behavior passed
hand-checks. The timed metric pair differed by at most 4.44e-15 absolute and
4.49e-15 relative, reflecting a different reduction order; scientific acceptance
under a numerical tolerance remains unselected. No 1024³ performance claim is
made from these smaller measurements.

The report, full derivative outputs, metric pairs, source snapshots and hashes
are in ignored `artifacts/benchmarks/staged-statistics/submission-ltHSVt/`.
The report confirms that both original source files remained unchanged.

## Full 1024³ JHS observations — job 24749966

The six derivative comparisons on DNS u and reconstructed u are bitwise
identical. These are single full-size calculation timings, excluding loading
and temporary-file output; they are not repeated warmed medians:

| Field / Python axis | Original | Slabbed |
| --- | ---: | ---: |
| DNS u / 0 | 13.66 s | 19.66 s |
| DNS u / 1 | 26.73 s | 31.62 s |
| DNS u / 2 | 28.63 s | 34.54 s |
| MPS u / 0 | 13.81 s | 55.91 s |
| MPS u / 1 | 26.95 s | 33.51 s |
| MPS u / 2 | 28.59 s | 92.68 s |

Worker peak memory fell from approximately 56 GiB to 16–17 GiB, but the slabbed
calculation was slower on every tested full-size axis. Retain it as an opt-in
memory-saving alternative; the smaller-case timing advantage does not establish
a production speed advantage. Neither version was replaced or modified.

For the DNS-u/MPS-u fidelity and relative L2 pair, chunking took 34.47 s versus
107.38 s (3.11× faster), with peak RSS 17364420 KiB versus 50916204 KiB (about
66% lower). Maximum relative difference was 1.55e-12. This reports numerical
agreement, not acceptance under a new tolerance. The reference pair is
fidelity 0.9982175306751173 and relative L2 0.04221930036336234, using the raw
kernel inputs without an additional mean subtraction.

The incremental full-size report is under
`artifacts/benchmarks/jhs-1024-uvw/submission-20p02r/comparison/report.json`.
The independent three-component DNS spectrum comparison is a separate stage in
that job; it does not affect these completed derivative/metric measurements.
