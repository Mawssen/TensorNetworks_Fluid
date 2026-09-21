# MPS v1 baseline plan

This is a validation plan only. It does not select a scientific dataset, establish tolerances, execute compression, or authorize a CRC submission.

## Phase 0: prerequisites

Before generating any numerical baseline:

1. Resolve the applicable items in [`docs/mps_v1_decisions.md`](../docs/mps_v1_decisions.md), especially bit significance, normalization, compression mode/value, environment, first real-data case, regression outputs, and tolerance policy.
2. Capture the historical runtime with `scripts/capture_mps_reference_environment.sh` from the same CRC context intended for the reference run. Preserve its text output with the baseline record.
3. Record checksums of the provisional source, wrapper, selected input, and eventually generated baseline artifacts.
4. Use a direct, single-case command. Do not submit the provisional sweep wrapper; its active array spans 28 cases ([`Julia/00mps_calc.slurm:11`](../Julia/00mps_calc.slurm#L11), [`Julia/00mps_calc.slurm:21-35`](../Julia/00mps_calc.slurm#L21-L35)).

## Phase 1: synthetic rank-one test

### Fixture definition

After the bit-significance decision is recorded, construct a small power-of-two three-dimensional tensor from independent two-entry factors for every axis bit:

```text
u[x, y, z] = product(a[axis, bit][selected_bit(x, y, z)])
```

Choose deterministic, finite, nonzero factor entries representable in the approved input dtype. Record every factor, tensor shape, dtype, index origin, axis order, bit significance, and whether normalization is applied. The exact values, size, dtype, bit convention, and normalization are **UNRESOLVED** until approved; do not silently choose them.

This construction is a product over the interleaved binary sites and therefore has an exact bond-dimension-one representation under the approved interleaved mapping. The test is intended to isolate tensorization, ordering, reconstruction, and serialization before testing lossy compression.

### Procedure

1. Materialize the fixture directly from the recorded bit/site definition; do not derive it from a scientific dataset.
2. Verify the fixture independently at each stage: original spatial tensor, binary reshape, literal split-1 permutation, flattened site vector, reconstructed binary tensor, and inverse-permuted spatial tensor. The provisional ordering and inverse are defined in [`Julia/0mps_calc.jl:181-217`](../Julia/0mps_calc.jl#L181-L217) and used for split 1 in [`Julia/0mps_calc.jl:317-361`](../Julia/0mps_calc.jl#L317-L361).
3. Exercise the approved no-loss/reference construction first. The provisional source represents its no-truncation branch as `cutoff=0` ([`Julia/0mps_calc.jl:381-384`](../Julia/0mps_calc.jl#L381-L384)); whether that branch is the approved oracle remains **UNRESOLVED** until the package environment and semantics are confirmed.
4. Record reconstructed values, output shape and dtype, `linkdims`, serialized keys/metadata, environment capture, command, elapsed time, and artifact checksums. The provisional reference obtains `linkdims`, contracts the MPS, and returns the reconstructed tensor at [`Julia/0mps_calc.jl:398-412`](../Julia/0mps_calc.jl#L398-L412).
5. Confirm that internal link dimensions are one for the exact rank-one construction and that forward/inverse ordering restores every element. A failure blocks real-data baselining.
6. Only after the exact case passes, exercise the single approved truncation mode/value and compare the same recorded observables.

### Acceptance criteria

- Structural checks may be exact: expected shapes, site count, permutation invertibility, output keys, and link-dimension pattern.
- Numerical tolerances are **UNRESOLVED**. Do not invent absolute, relative, fidelity, or norm thresholds until the historical Julia/package/BLAS environment and observed output dtype have been captured and the fidelity/error definition has been approved.
- Any unexpected discrepancy must be reported as a discrepancy; do not classify it as a historical bug or intended behavior without a decision.

## Phase 2: later real-data baseline

This phase starts only after the synthetic test passes and the user selects the dataset, timestep, field, crop/layout, compression mode/value, outputs, and resource envelope.

1. Record the selected input's immutable path or dataset identifier, checksum, byte size, field, timestep, units, full dimensions, endian convention, dtype, axis convention, and crop.
2. Capture the environment immediately before the reference run and retain the Slurm allocation details if run within an allocation.
3. Run the provisional numerical reference once with the approved single-case arguments. Do not submit an array or sweep.
4. Preserve stdout/stderr, exit status, command, wall time, peak-memory information if available without changing the approved resources, output files, and checksums.
5. Inspect and record MAT variable names, shapes, dtypes, summary statistics, link dimensions, and approved error/fidelity metrics. The provisional writer uses the keys `u` and `chi_crit` ([`Julia/0mps_calc.jl:416-425`](../Julia/0mps_calc.jl#L416-L425)).
6. Run the future clean implementation on the identical input in the captured compatibility environment, then compare only the approved regression outputs using the approved metrics.
7. If results differ, localize the first divergent stage—decode, crop, tensorization, permutation, MPS construction, truncation, reconstruction, or serialization—before asking whether the difference is a bug or intended behavior.
8. Promote artifacts to a baseline only after the environment, inputs, outputs, metrics, tolerances, and discrepancy disposition are documented and approved.

No numerical tolerance is specified in this plan because neither the authoritative historical environment nor the reference output dtype has yet been established.
