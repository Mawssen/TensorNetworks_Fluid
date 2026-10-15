# MPS v1 dry-run specification

Implementation status, 2026-10-01: the owner authorized a narrower first [declaration-only interface](mps_v1_dry_run_interface.md) using an [explicit JSON template](mps_v1_dry_run_template.json). It supports chi tokens and rejects numeric cutoffs with Q04; this temporary limitation does not change the historical calculation. It does not access input paths or write proposed artifacts. Filesystem, runtime, CRC, approval and diagnostic collection checks below remain requirements for a future complete CRC plan, not claims that the first implementation satisfies them. Its success status is explicitly `DECLARATIONS VALID; EXECUTION PREFLIGHT UNVERIFIED`.

Status: documentation, not an implemented command. Read with the [approved plan](mps_automation_plan.md), [question register](mps_v1_open_questions.md), and [commented template](mps_v1_config_template.toml). Unresolved Q IDs are gates, not permission to infer a value. No wrapper code is authorized by this specification.

## Execution boundary

The future planning command defaults to dry-run and **never calls `sbatch`**, including `sbatch --test-only`. It never runs the historical Julia script, Julia formatting probes, module setup, package installation, or scheduler queries. Validation is limited to configuration text, approved source/provenance files, and read-only filesystem metadata under the selected Q11/Q17 policy. It does not load raw fields or compute scientific metrics. The future dry run may create new audit files under an explicitly approved audit directory; it must not create historical result directories or overwrite anything. This present milestone creates documentation only.

Q16 must freeze the command name/parser/interface. A missing decision produces a blocked report, not a runnable plan. An entirely commented template parses as an empty document and must be rejected as a case configuration. Every `REQUIRES_OWNER_DECISION` token is unresolved.

## Required validation

1. Parse exactly one configuration with the agreed schema. Reject malformed syntax, duplicate/unknown keys, absent required declarations, and multiple-case selectors. Fixed shape/axis/bounds tuples are not case arrays. Validate scalar types without silently coercing timestep strings or mode tokens.
2. Require explicit nonempty dataset identifier, field, timestep, absolute input path, and unit string. Never choose any from a filename roster or the historical Slurm array. Confirm the path is a readable regular file in the verified execution context. Resolve symlinks only under Q17 while preserving the basename passed to Julia. Match the explicit timestep, including leading zeros, against the CLI's `_(digits).dat` basename suffix ([source lines 821–824](../Julia/0mps_calc.jl#L821-L824)). Units are recorded verbatim for owner approval; no registry is consulted.
3. Match the approved input contract exactly: Jet Flame DNS; `Float32`; big-endian; `(x,y,z)` shape `(864,1008,576)`; `x` fastest; centered `(512,512,512)` crop before tensorization; zero-based half-open `x=[176,688)`, `y=[248,760)`, `z=[32,544)`. The [source crop, lines 838–842](../Julia/0mps_calc.jl#L838-L842) corresponds to Julia one-based inclusive `177:688`, `249:760`, `33:544`. These are validation declarations; the wrapper performs no cropping itself. File size cannot prove endianness or physical units.
4. Print observed input size versus required read size 2,006,581,248 bytes. Reject shorter input. Longer input remains blocked until Q01 resolves exact-size versus trailing-byte acceptance. Apply only the approved Q11 identity policy; do not silently hash the dataset.
5. Require exactly one ordering integer from 1–4 and its Q02-approved label. Map only the evidenced filename tags: `1=il`, `2=seq`, `3=comb1`, `4=combn` ([source lines 852–856](../Julia/0mps_calc.jl#L852-L856)). Reject 5/6 even though helper functions exist. No default ordering or inferred bit meaning.
6. Require exactly one cutoff or maximum-bond-dimension mode and one value. Parse according to [source lines 811–819](../Julia/0mps_calc.jl#L811-L819), then enforce the owner-selected Q03 domain. A cap must be representable as the target Julia integer. Reject ambiguous modes, multiple values, parse failures, and values outside that approved domain. Numeric-domain uncertainty is a blocker, not an invented positive-only rule.
7. Verify the absolute historical script path identifies the unchanged `Julia/0mps_calc.jl` for the approved commit/hash and Q12 dirty-state policy. Validate the explicit Julia executable and represented Q07 environment without executing either. Existence does not prove that packages load; show that distinction.
8. Resolve absolute working directory, expected output, audit directory, and log destinations. Check existence/permissions using the agreed Q06/Q17 policy without creating a result directory or write-probing historical paths. Reject collisions with input, source, existing output, or other artifacts. Validate Q14 launch-time rechecks for the interval after planning; a planning check alone cannot guarantee no overwrite later.
9. Validate all CRC declarations: cluster, partition, account/QOS/reservation/constraint values or explicit approved unset state; node/task/CPU counts; memory with units/request semantics; walltime; job name/log targets. Reject malformed directives, newline/control-character injection, inconsistent declared threading, and any array/sweep/replicated-payload request. The selected Q08/Q09 rules define permissible values. Never claim allocation eligibility, capacity, or queue time has been verified offline.
10. Validate exact quoting of argv, working directory, environment, and generated scheduler directives under the selected runtime/shell contract. Do not evaluate configuration text as shell code. Reject unsupported escaping or expansions instead of interpolating unsafe text. Setup commands require explicit approved representation under Q07/Q16 and are printed, not executed.
11. Require the structural-only diagnostic scope and agreed Q15 reader/context/artifact destinations. Reject fidelity/error, scientific normalization/filtering/unit conversion, reference comparisons, automatic acceptance, or follow-on submission options. If Q15 is unresolved, a report may explain the blocker but cannot assert a complete job/handoff plan.

## Exact command and output

Print both an argv list and a shell-quoted human-readable equivalent, along with cwd and environment. Conceptually the one payload is:

```text
<absolute-julia-executable> <absolute-repository>/Julia/0mps_calc.jl <absolute-input-path> <mode-token> <ordering-integer>
```

The actual report must replace every placeholder. Any approved runtime flags must also appear exactly; unknown flags/environment choices block a complete command. Print setup and launcher choices separately without executing them. There must be exactly one historical calculation invocation.

For cutoff mode, the output label is `cf` plus Julia's string representation of the parsed `Float64`, not necessarily the supplied token. For cap mode, pass `chi=<integer>` and use `chi<parsed-integer>`. Follow Q04; inability to prove the cutoff label blocks output-path certification. No guessed conversion or arbitrary user filename override is allowed.

Print the historical filename and full absolute path separately:

```text
truncated_<input-basename>_512_<tag>_<label>.mat
<working-directory>/truncated_<timestep>/truncated_<input-basename>_512_<tag>_<label>.mat
```

The input basename includes `.dat`. No extra ordering directory, renamed file, altered MAT payload, or source copy is introduced. Q06 may specify a new working directory while retaining the exact relative naming algorithm ([source lines 858–866](../Julia/0mps_calc.jl#L858-L866)).

## Mandatory printed report

Every report, even a blocked one, prints known values and explicitly identifies unknown/unverified fields:

- State: `DRY RUN — NOT APPROVED — NOT SUBMITTED`, plus `COMPLETE FOR OWNER REVIEW` or `BLOCKED` and all Q IDs/rejection reasons. Complete never means approved or scientifically acceptable.
- Case identity, unit string, exact input path/basename/timestep, all declared/expected input-contract fields, observed bytes, input-identity evidence, and verification context.
- Exact argv and quoted command, cwd, script identity, Julia executable, explicit environment/setup/launcher and thread settings. Internal historical BLAS threads remain 10; requested CPUs are separately displayed.
- Exact historical filename and absolute MAT path, collision status, stdout/stderr paths, audit and diagnostic destinations; expected MAT variables `u` and `chi_crit`.
- Every requested Slurm field and explicit unset value; one non-array job, no sweep/concurrency/default campaign. Show the full proposed directive block. Requested limits are not performance estimates.
- Scope: one field/file/order/mode, 501,645,312 raw values (2,006,581,248 bytes), 134,217,728 retained values (512 MiB if Float32), 27 qubit sites, one MPS construction, dense reconstruction, and MAT write. These array sizes are not peak-memory estimates. Show diagnostic/preflight I/O separately under Q11/Q15.
- Runtime/peak-memory estimates with cited retained evidence, or `unknown`; no timing benchmark, extrapolated resource choice, queue prediction, or scientific metric.
- Provenance: full calculation/planner commits, source SHA-256, dirty-state record, exact original configuration and hash, separately canonicalized configuration and hash, generated job hash, digest algorithm/coverage, plan digest, and recorded environment/identity evidence. Missing required provenance blocks certification. Q16 defines serialization to avoid a self-referential digest.
- Approval still required, exact field/unit pairing for review, and the required future post-run summary of structural observations and all artifact locations. No runnable submission command is offered before approval.

## Rejection and completion semantics

Every failed required check above blocks the plan; multiple failures are accumulated when safe rather than silently defaulted. Unsupported scientific/CRC selections, unverified filename formatting, placeholders, missing provenance, unavailable filesystem checks, existing outputs, injection hazards, and unresolved validation policies are explicit blockers. Report unreadable files without modifying them. Record the limits of stat/permission checks and owner-supplied metadata.

Blocked preparation cannot produce an artifact represented as submission-ready. A complete plan still needs explicit owner approval bound to its exact digest, then the owner's separate manual submission. Q13/Q14 govern invalidation and launch-time verification. The future wrapper never calls `sbatch` automatically or otherwise; scheduler query/collection permissions are separate and do not belong to dry-run.

After an approved real run, report only the approved structural/run diagnostics and their sources, including full `chi_crit`, with exact output/log/diagnostic locations. Missing accounting, exit state, timing, or MAT metadata stays unavailable. Neither successful scheduling nor a readable MAT file proves scientific correctness. The owner reviews the result.
