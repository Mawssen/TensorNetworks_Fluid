# Single-case execution

The execution stage authorized in the current session uses the existing JSON
schema, validation, argv construction, cwd, and historical output naming.
Dry run remains available through `tools/mps_v1_dry_run.py`.

Within an existing CRC compute allocation, after the owner approves the case:

```sh
python3 -B tools/mps_v1_run.py /absolute/path/to/approved-case.json
```

The runner invokes `legacy/Julia/0mps_calc.jl` directly, with the planner's exact argv,
the declared working directory, and declared OMP/Julia thread settings. `unset`
removes JULIA_NUM_THREADS from the child environment. Other environment values,
including project/depot/startup and already loaded modules, are inherited.
Module declarations are recorded, never evaluated as shell code; prepare the
approved environment before invoking the runner. No resource allocation or
submission is performed. A Slurm job ID is required and login hosts are rejected.

Routine engineering choices: the working directory must exist; log parents are
created and log files are opened exclusively; audit.directory must be new.
Existing expected output, logs, or audit directories block execution. Run once
per target directory: the existence check is not a cross-process lock against
unrelated historical writers. The runner does not choose an input length policy,
hash datasets, restrict chi beyond the existing validator, or certify a complete
CRC preflight. Those open questions remain in the register. Use only explicitly
approved case/configuration/environment values.

The audit directory contains the original/canonical configuration, dry-run plan
and command summary, plus `run.json` with actual execution state, exit code,
elapsed time, output existence/size, and log locations. The stored dry-run plan
retains its original proposal state; it is not an approval record. Nonzero child
exits propagate; launch errors and missing output after a zero exit fail. MAT
contents and Slurm accounting are marked unavailable rather than inferred.

The historical Julia source now wraps the CLI in `mps_cli` and extracts its
binary decoding/center crop into `load_centered_be_float32`. The CLI uses that
loader's original grid/crop defaults. A direct-execution guard allows inclusion
without starting the DNS calculation. Numerical compression, ordering,
reconstruction, and output-writing implementations remain unchanged. The
documented public API is exported by `src/TurbulenceTools.jl` through
`using TurbulenceTools` with `--project=.`; see the root README for
synthetic cases and the prepared historical comparison.

`tools/mps_v1_smoke.jl <chi> <ordering>` exercises these functions on a 2×2×2
Float32 tensor of ones within a compute allocation. It checks shape, finite
values, and rank-one link dimensions, without fidelity thresholds, data reads,
or MAT writes. This tests function integration, not the production binary CLI.
Example diagnostic arguments `1 1` are synthetic test parameters only.

Lightweight tests:

```sh
python3 -B -m unittest discover -s test/python -p 'test_mps_v1*.py' -v
```

Execution tests launch a fake executable, checking actual argv, cwd, threads,
logs, provenance, failures, and collisions. They do not validate MPS numerics.
