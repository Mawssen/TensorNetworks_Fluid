# First performance optimization: Float64 spectrum binning

The owner approved Float64 binning as the preferred spectrum default after the
full-size cross-check. Use `turbulence_tools_stats.calculate_spectrum`; request
`implementation="reference"` to recover the original. The explicit
`calculate_spectrum_binned` entry point remains available.
The archived historical workflows and sources remain preserved:

- `legacy/mps_py_codes/calculate_spectrum.py`: preserved Float64 reference.
- `legacy/py/jhs_velocity_spectrum_stats.py`: preserved separate Float32 variant.
- `src/python/turbulence_tools_stats/spectrum.py`: new Float64 binned version.

Keeping this stage in Python reuses the existing NumPy FFT backend without new
dependencies or a language/FFT migration. Import from a case script as follows:

```python
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]  # for cases/<category>/case.py
sys.path.insert(0, str(root / "src/python"))
from turbulence_tools_stats import calculate_spectrum_binned

# Explicitly supplied, same-shaped cubic Float64 velocity components.
k, spectrum = calculate_spectrum_binned(u, v, w)
```

This does not require or provide a Julia/Python bridge. The existing
`using TurbulenceTools` API and historical 2FI helper remain unchanged.

## What changes

The old method masks/scans the full volume separately for each shell. The new
method assigns each Fourier sample to the same shell and accumulates weighted
sums with bincount. It processes one component at a time and builds geometry in
bounded slabs, avoiding retention of three power cubes and full-volume geometry.
The full 3D FFT is still required; this is not an out-of-core FFT or a GPU port.

Input axes, the 2*pi domain, FFT power normalization, historical wavenumber order,
digitize right=True boundaries, zero/overflow exclusion, and final shell-count
normalization are preserved. There is no implicit mean subtraction. Components
and all floating intermediates remain Float64 (Complex128 FFTs). Integer shell
labels/counts are kept separately. The new entry point requires Float64 arrays
and rejects other dtypes rather than silently converting them.

Weighted accumulation changes floating-point reduction order. Neither bitwise
equality for general fields nor scientific acceptance under an unspecified
tolerance is claimed. The comparison records exact equality and maximum
absolute/relative differences; relative differences exclude zero reference
values and zero-reference mismatches are reported separately. Shell labels,
counts, wavenumbers, and hand-checkable examples are tested exactly.

## Cross-check and benchmark

Inside an approved compute allocation, from the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -B cases/benchmarks/spectrum_binning.py \
  --output-directory artifacts/benchmarks/spectrum-binning/NEW_RUN \
  --sizes 32 64 128 --repetitions 3
```

The output directory must be new. This command uses deterministic synthetic data
only. It runs tests, then all three implementations in independent processes.
Timing is the median of warmed full-function calls (including FFTs). Linux peak
process RSS includes inputs, workspace, and warmup. It also saves the original
source snapshots, hashes, per-worker measurements, complete spectra, software
versions, and a JSON difference/timing report. The Float32 version is measured
as a separate implementation, not substituted for the Float64 reference.

`cases/benchmarks/spectrum_binning.slurm` prepares one CPU, 4 GiB and 10 minutes
on the documented smp/smp, pgivi, short-QoS setup. Submission needs owner approval.
No job is submitted by the Python script. Numerical tests skip on login nodes.

This comparison does not authorize running full 1024³ spectra, producing the
missing compressed v/w fields, or replacing production statistics defaults.
Those remain separate execution/scientific decisions.

## First CRC measurement — 2026-11-16

Job `24749888` completed with exit `0:0` on `smp-n221`, using one CPU. All four
test methods passed, including exact shell/count checks and hand-checkable
spectra. The warmed full-function median timings (three repetitions) were:

| Cube | Original Float64 | New Float64 | Existing Float32 variant | Float64 speedup |
| --- | ---: | ---: | ---: | ---: |
| 32³ | 3.597 ms | 2.021 ms | 9.355 ms | 1.78× |
| 64³ | 31.184 ms | 19.216 ms | 23.128 ms | 1.62× |
| 128³ | 362.541 ms | 192.699 ms | 187.963 ms | 1.88× |

At 128³, separate-worker peak RSS was 265180 KiB for the original versus
172008 KiB for the new Float64 version (about 35% lower). The maximum relative
difference on nonzero reference shells was 7.90e-16 for the new version, versus
1.02e-8 for the existing Float32 variant. Across all three sizes the new
version's maximum absolute difference was 2.78e-17. Random-field results are
not bitwise identical. These are measured differences, not an owner-approved
scientific tolerance or a guarantee for 1024³ inputs.

Full spectra, timing samples, source snapshots/hashes, test logs, and the report
are saved locally under ignored
`artifacts/benchmarks/spectrum-binning/submission-NlqLwh/`. The report verified
that both original source files remained unchanged. No production default was
switched and no production dataset was read by this comparison.

## Full 1024³ DNS U/V/W follow-up

The separately approved real-data job `24749966` completed all comparisons.
Historical Float64 binning took 1153.92 s versus 174.83 s for the new Float64
kernel (6.60× faster). Worker peak RSS fell from 88.6 to 56.6 GiB. Maximum
relative spectral difference was 2.64e-15; wavenumbers were identical and all
spectra were finite. These are single-call timings excluding loading/output.
The preserved Float32 variant took 158.57 s, with a larger maximum relative
difference of 9.73e-9 and 64.6 GiB worker peak RSS. Details and scope are in
[the full-size comparison record](jhs_full_crosscheck.md). Defaults remain
unchanged; no scientific acceptance tolerance has been selected.
