# Safe CRC automation plan for one historical MPS case

## First-interface implementation update — 2026-10-01

The owner has authorized the first executable declaration-only interface described in [mps_v1_dry_run_interface.md](mps_v1_dry_run_interface.md), with its [explicit JSON template](mps_v1_dry_run_template.json). It prints a proposal without reading or statting DNS input, executing Julia, creating artifacts, or interacting with Slurm. This limited implementation supersedes earlier documentation-only restrictions for this stage; the full CRC preparation/approval design below remains future work.

The owner explicitly selected **chi-only support with numeric cutoff rejection while Q04 remains unresolved**. No cutoff-label table is approved. This is a temporary interface limitation, not a change to the historical workflow. Integer tokens and literal historical ordering tags are validated without certifying scientific cap suitability or a semantic bit convention. Resource and environment declarations are displayed; remaining preflight gates are explicitly unverified. Success means valid declarations, not a complete or approved CRC plan.

## Status and boundary

This is a design document only. It does not select a scientific case, endorse a CRC resource request, or authorize execution. The proposed wrapper is a thin planning and launch-description layer around the unchanged active command-line path in [`Julia/0mps_calc.jl`](../Julia/0mps_calc.jl). [`Julia/00mps_calc.slurm`](../Julia/00mps_calc.slurm) is evidence of a historical CRC invocation, not a template whose active campaign or resource values are silently inherited.

**Decision recorded 2026-09-21:** For version 1, only the active three-argument CLI in `Julia/0mps_calc.jl` is the authoritative numerical calculation. The wrapper will invoke that path unchanged. No other historical MPS script is part of the v1 numerical contract. The conflicting comments and active arrays in `Julia/00mps_calc.slurm` do not establish a production field roster; field, dataset, timestep, ordering, truncation, CRC resources, and output location remain explicit user-approved configuration decisions.

**Decision recorded 2026-09-21:** Version 1 is case-agnostic. Every configuration must explicitly provide exactly one dataset identifier, field, timestep, and absolute input path. The wrapper must not hard-code, discover, or infer any of those values from `Julia/00mps_calc.slurm`, its historical array, directory contents, filename catalogs, or defaults. No production case must be selected to implement the generic wrapper.

**Decision recorded 2026-09-21:** Every version 1 configuration must explicitly declare scalar type, endianness, full-grid shape, centered crop, axis convention, and physical units. The wrapper must fail closed if a fixed input-contract declaration is absent or mismatched. Values established directly by the CLI are big-endian `Float32`, full-grid shape `(864,1008,576)`, and a centered `(512,512,512)` crop. The following records resolve physical-axis semantics and the separate user-approved units policy.

**Decision recorded 2026-09-21:** The v1 input domain is Jet Flame DNS. Its physical array-axis convention is `(x,y,z)=(864,1008,576)`, with `x` fastest-varying in the raw binary stream. Before tensorization and compression, the unchanged historical calculation takes the centered `(512,512,512)` crop. Expressed as zero-based half-open bounds, that crop is exactly `x=[176,688)`, `y=[248,760)`, and `z=[32,544)`. Every configuration must assert this domain, axis convention, fastest-varying axis, crop shape, and crop bounds; the wrapper must reject a mismatch.

**Decision recorded 2026-09-21:** Every configuration must contain a nonempty physical-unit string. Version 1 will not contain a field-to-unit registry and will not infer units. The wrapper validates presence and records the exact string; the owner's explicit approval of the plan certifies the field/unit pairing for that case.

**Decision recorded 2026-09-21:** After any approved real run, Codex must present the owner with a concise results summary and the exact locations of the reconstructed output, logs, and computed diagnostics. Scientific review and acceptance belong to the owner.

**Approved diagnostic scope:** Version 1 reports only Slurm state/accounting, exit state, elapsed time, output existence and size, MAT variable names/shapes/dtypes, and `chi_crit`. It must not compute or interpret scientific fidelity/error metrics. Unavailable observations are reported as unavailable, never guessed. This documentation task does not authorize querying Slurm or reading result files.

The [open-question register](mps_v1_open_questions.md) is the single current list of unresolved decisions. The [commented configuration template](mps_v1_config_template.toml) and [dry-run specification](mps_v1_dry_run_spec.md) describe the remaining work without choosing those decisions. The owner is unavailable: record blockers there and do not request further answers during this documentation milestone.

This milestone creates documentation and a fully commented, non-executable configuration template only. It creates no runnable configuration or wrapper implementation and changes no historical source, environment, dataset, job, or result.

The automation target is deliberately one case:

- exactly one semantic field;
- exactly one input file path and timestep;
- exactly one ordering code;
- exactly one truncation mode and value; and
- exactly one non-array Slurm job description.

Lists, globs, directory discovery, field inference, mode sweeps, ordering sweeps, job arrays, and chained submissions are out of scope.

## Evidence from the provisional references

The active Julia path accepts three positional arguments: `input_path`, a mode string, and integer `sp` ([`0mps_calc.jl:806-809`](../Julia/0mps_calc.jl#L806-L809)). A case-insensitive `chi=<integer>` selects a maximum bond dimension; every other mode string is parsed as a `Float64` cutoff ([`0mps_calc.jl:811-819`](../Julia/0mps_calc.jl#L811-L819)). The active path then:

1. requires an input basename ending in `_<digits>.dat` and uses those digits as the timestep ([`0mps_calc.jl:821-824`](../Julia/0mps_calc.jl#L821-L824));
2. reads big-endian `Float32` words for a hard-coded `(864, 1008, 576)` grid and takes a centered `512^3` crop ([`0mps_calc.jl:825-842`](../Julia/0mps_calc.jl#L825-L842)); under the approved Jet Flame DNS convention this is `(x,y,z)` with `x` fastest-varying, and the zero-based half-open crop bounds are `x=[176,688)`, `y=[248,760)`, and `z=[32,544)`;
3. applies the selected ordering and MPS truncation without an explicit normalization step in the active calculation ([`0mps_calc.jl:366-394`](../Julia/0mps_calc.jl#L366-L394), [`0mps_calc.jl:846-850`](../Julia/0mps_calc.jl#L846-L850)); and
4. writes MAT variables `u` and `chi_crit` ([`0mps_calc.jl:416-425`](../Julia/0mps_calc.jl#L416-L425)).

The output directory is relative to the process working directory. It is `truncated_<timestep>`. The output basename is:

```text
truncated_<input-basename>_512_<ordering-tag>_cf<Julia-parsed-cutoff>.mat
```

or:

```text
truncated_<input-basename>_512_<ordering-tag>_chi<integer>.mat
```

The active filename tags are `1 -> il`, `2 -> seq`, `3 -> comb1`, and `4 -> combn` ([`0mps_calc.jl:852-866`](../Julia/0mps_calc.jl#L852-L866)). Although ordering helpers for codes 5 and 6 exist, the active filename path rejects them, so the proposed wrapper must not offer them.

The provisional Slurm file records one node, one task, 20 CPUs, `smp` cluster and partition, 64 GiB, two hours, and a 28-element array ([`00mps_calc.slurm:2-11`](../Julia/00mps_calc.slurm#L2-L11)). It also records `OMP_NUM_THREADS=10`, a Julia 1.12.6 executable path, two truncation choices, and ordering code 1 ([`00mps_calc.slurm:18-19`](../Julia/00mps_calc.slurm#L18-L19), [`00mps_calc.slurm:29-31`](../Julia/00mps_calc.slurm#L29-L31)). These are historical values only.

There is a material internal mismatch in that file: its prose says mixfrac and four tasks, while its active arrays specify 14 other fields, two modes, and 28 tasks ([`00mps_calc.slurm:13-16`](../Julia/00mps_calc.slurm#L13-L16), [`00mps_calc.slurm:21-35`](../Julia/00mps_calc.slurm#L21-L35)). The wrapper must not resolve that mismatch by choosing either side.

## Minimal wrapper design

### Proposed interface and artifacts

The eventual implementation should be a dependency-free planning command, using only a standard-library configuration parser. A single invocation would conceptually be:

```text
mps-case-plan <one-case-config>
```

Dry-run is the default and the only behavior of the planning command. It performs read-only validation and writes, to a new audit directory rather than the historical result directory:

- an exact original configuration snapshot and a separately canonicalized configuration record, each with its own hash;
- `plan.json`, containing resolved paths, provenance, preflight facts, estimated scope, approval state, and a digest;
- `case.slurm`, containing exactly one job and no array directive; and
- `plan.txt`, the human-readable dry-run report.

The planner contains no `sbatch` call and no code path that can invoke `sbatch`. It does not execute Julia. This separation makes accidental submission impossible from the preparation command.

The generated Slurm script is a transport wrapper only. It must not include, copy, patch, or reimplement Julia numerical logic. Its payload is the exact equivalent of:

```text
cd -- <approved-working-directory>
export OMP_NUM_THREADS=<approved-value>
<approved-julia-executable> <repository>/Julia/0mps_calc.jl <one-input-path> <one-mode-argument> <one-ordering-code>
```

Every token must be rendered with shell-safe quoting in the dry-run report and generated script. All paths shown in the plan must be resolved absolute paths, while `basename(input_path)` must remain unchanged so the historical result name remains unchanged.

### One-case configuration contract

Case selectors must be single values; fixed axis, shape, and crop tuples are allowed. Placeholder text below is a schema description, not a proposed case. The complete documentary field inventory is in the [template](mps_v1_config_template.toml); schema choices still needing approval are in the [register](mps_v1_open_questions.md).

```text
case.dataset_id                 required descriptive identifier
case.field                      required one-field identifier
case.timestep                   required string, preserving leading zeros
case.input_path                 required absolute path to one .dat file
case.input.domain               required assertion: Jet Flame DNS
case.input.scalar_type          required assertion: Float32
case.input.endianness           required assertion: big-endian
case.input.axis_order           required assertion: (x,y,z)
case.input.full_grid_shape      required assertion: (864,1008,576) in (x,y,z) order
case.input.fastest_axis         required assertion: x
case.input.crop_kind            required assertion: centered
case.input.crop_shape           required assertion: (512,512,512)
case.input.crop_bounds          required zero-based half-open assertions:
                                x=[176,688), y=[248,760), z=[32,544)
case.input.physical_units       required nonempty user-approved unit string
case.ordering_code              required one of the approved active codes 1..4
case.ordering_label             required human-readable assertion
case.truncation.kind            required: cutoff OR max_bond_dimension
case.truncation.value           required one value; allowed domain unresolved (Q03)

historical.julia_script         exact path to Julia/0mps_calc.jl
historical.working_directory    directory that determines historical output location
runtime.julia_executable        exact executable path
runtime.omp_num_threads         exact positive integer
runtime.julia_num_threads       exact value or explicit "unset"
runtime.module_environment      exact approved module/setup commands, possibly empty

slurm.cluster                   explicit value
slurm.partition                 explicit value
slurm.account                   explicit value or explicit "unset"
slurm.qos                       explicit value or explicit "unset"
slurm.nodes                     explicit value
slurm.ntasks                    explicit value
slurm.cpus_per_task             explicit value
slurm.memory                    explicit value with unit
slurm.walltime                  explicit value
slurm.job_name                  explicit value
slurm.stdout_path               exact non-colliding path
slurm.stderr_path               exact non-colliding path or explicit join with stdout

audit.directory                 exact path outside the historical result directory
approval.required_approver      repository-owner identity or other agreed identity
```

The schema must reject unknown keys, absent values, multiple input paths, both truncation modes, unparseable values, and ordering codes not supported by the active output-name path. The earlier proposed positive-only truncation rule is not an owner decision: Q03 must resolve the accepted numeric domain, including zero cutoff, before that validator is written. Fixed input-contract mismatches are rejected. The physical-unit value must be a nonempty string, preserved exactly in the plan and approval record; the wrapper does not infer or compare it with a registry. Fixed axis, shape, and bounds tuples are allowed; campaign arrays remain forbidden. Nothing receives a scientific or CRC default.

Dataset identifier, field, and timestep are descriptive assertions supplied by the user; they must not be inferred from the input path or historical Slurm content. The planner must separately check that the explicitly supplied timestep agrees with the `_<digits>.dat` suffix required by the authoritative Julia CLI, because that CLI uses the suffix to construct the output path. A mismatch must fail rather than silently replace either value. The absolute input path must exist and be readable at planning time. Its observed byte size and the historical read requirement must be shown; the accept/reject policy for trailing bytes remains a decision below.

### Exact historical command and output derivation

The planner must deterministically derive the Julia argument and expected result path, then display both before approval:

- `cutoff`: pass the approved numeric token as argument 2; derive the output label using semantics identical to Julia's `string(parse(Float64, token))`, because the active code names the file from the parsed value rather than necessarily preserving the input spelling;
- `max_bond_dimension`: pass `chi=<approved-integer>` and derive `chi<integer>`; and
- ordering: pass the approved integer as argument 3 and use the active `il`, `seq`, `comb1`, or `combn` tag.

The expected output is resolved as:

```text
<historical-working-directory>/truncated_<timestep>/
  truncated_<input-basename>_512_<tag>_<mode-label>.mat
```

The planner must fail closed if it cannot reproduce Julia's cutoff-string formatting exactly. It must also fail if the expected result already exists. The first implementation should have no overwrite flag: replacing a historical result would require a separately approved policy change.

The MAT file itself must retain the historical filename and contents. Provenance belongs in the separate audit directory, not in, beside, or inside the historical output unless a later decision explicitly changes that policy.

### Mandatory dry-run report

Before approval, `plan.txt` and terminal output must show all of the following in one reviewable block:

1. **Case:** dataset identifier, field, timestep, exact input path, all declared and expected input-contract values, observed input size, ordering code/label/tag, and truncation kind/value.
2. **Execution:** exact working directory; exact, shell-quoted Julia command; Julia executable; historical script path and SHA-256; relevant thread variables and environment setup.
3. **Output:** exact expected MAT path, its two historically written variable names, and an explicit exists/absent collision check.
4. **Resources:** cluster, partition, account/QOS state, nodes, tasks, CPUs per task, memory, walltime, job/log names, and confirmation that this is one non-array job.
5. **Estimated scope:** the evidence-based scope described below, plus any owner-supplied runtime or memory estimate and its source. A Slurm request must be labeled a limit/request, not an observed estimate.
6. **Provenance:** full Git commit, repository dirty-state summary, SHA-256 of the historical Julia file, canonical configuration and its SHA-256, generated Slurm SHA-256, plan digest, and planner version/commit.
7. **Safety state:** `DRY RUN — NOT APPROVED — NOT SUBMITTED`, with every unresolved decision listed. No submission command is presented as runnable while this state holds.

### Evidence-based scope estimate

From the active constants, one case attempts to read exactly `501,645,312` `Float32` values, or `2,006,581,248` bytes (about 1.869 GiB), before retaining `512^3 = 134,217,728` values (512 MiB at `Float32`) ([`0mps_calc.jl:825-842`](../Julia/0mps_calc.jl#L825-L842)). A three-dimensional `512^3` tensor produces 27 qubit sites under the active tensorization. The case performs one MPS construction/truncation, one full reconstruction, and one MAT write.

The references provide no observed peak memory, elapsed time, package-resolved performance record, or completed-job accounting. Therefore the dry run must initially report peak memory and runtime as **unknown**, not infer them from the provisional 64 GiB/two-hour request. If the owner later supplies a comparable retained job record, the plan may show that estimate with its provenance and uncertainty.

### Approval and manual submission gate

Approval is a separate, non-submitting action after the owner reviews the complete dry-run plan. It records:

- approver identity;
- UTC approval time;
- exact plan digest;
- the statement that the displayed case, command, output, resources, and scope are approved; and
- an optional expiration time.

Approval is invalidated by any change to the configuration, Git commit or historical script bytes, generated Slurm bytes, input identity/size policy, expected output path, working directory, runtime environment, or resource request. The approval action still does not call `sbatch`.

The approval statement must include the exact field and physical-unit string. This is the case-specific scientific authorization for that pairing; no hard-coded registry substitutes for it.

Only after a valid approval record exists may the tool print the exact manual submission command, for example the resolved equivalent of `sbatch <case.slurm>`. The owner must run that command separately. The wrapper never submits automatically, never submits as a side effect of planning or approval, and never creates dependencies or follow-on jobs.

Immediately before manual submission, a read-only `verify` action must recompute the plan digest, recheck that the input is readable and the output absent, and report either `APPROVED AND UNCHANGED` or `BLOCKED`. Verification does not submit.

### Mandatory post-run review handoff

After an approved real run finishes, the automation must preserve the historical output unchanged and assemble a review handoff. Codex must show the owner:

- a concise factual run summary, including job identity and observed completion state;
- the exact reconstructed MAT output path;
- the exact stdout and stderr log paths;
- the exact paths of any diagnostic artifacts and a concise presentation of the computed diagnostic values; and
- any missing, unreadable, or unexpected artifact as an unresolved issue.

The report must distinguish observed facts from scientific judgments. It must not label the result scientifically accepted, valid, faithful, converged, or satisfactory. The approved diagnostic set is Slurm state/accounting, exit state, elapsed time, output existence/size, MAT variable names/shapes/dtypes, and `chi_crit`. Report each observation's source; keep Julia elapsed time and Slurm elapsed time separately labeled. Preserve the full observed `chi_crit` sequence in the diagnostic artifact. Reader, artifact format, and collection-context decisions remain in Q15. No fidelity/error metrics or scientific acceptance thresholds belong in v1.

## Remaining decisions and implementation gate

All unresolved scientific, compatibility, CRC, and implementation choices are consolidated in [mps_v1_open_questions.md](mps_v1_open_questions.md). Its stage labels distinguish decisions blocking validator implementation or a complete dry-run plan from later per-case submission choices and out-of-scope scientific validation. No unresolved option is selected here. No code is authorized in this milestone.

Earlier numerical-rebuild documents are retained as background, not as additional v1 requirements. V1 invokes the historical calculation unchanged and adds no normalization, filtering, unit conversion, fidelity calculation, or numerical regression campaign. Explicit case choices remain required later; they do not require selecting a production case to design the generic schema.

## Implementation acceptance criteria

Implementation is complete only when non-executing tests demonstrate that:

- a valid configuration can describe only one explicit dataset/field/timestep/absolute-path/ordering/truncation case;
- every configuration explicitly declares the full input contract and any absent or mismatched declaration fails closed;
- the Jet Flame DNS axis order, fastest-varying axis, crop shape, and zero-based half-open crop bounds are matched exactly;
- the exact nonempty unit string is bound to the case approval without a field registry;
- omitted scientific or CRC values fail rather than default;
- dry-run displays the exact command, output, resources, evidence-based scope, and unknown estimates;
- cutoff and `chi=` output names exactly match the active Julia naming semantics;
- existing outputs, unsupported orderings, ambiguous truncation, arrays, globs, and config drift are blocked;
- provenance contains the Git commit, dirty state, script/config/generated-file hashes, and plan digest;
- approval is bound to that digest and any change invalidates it;
- preparation, approval, and verification cannot invoke `sbatch` or Julia; and
- the generated job contains one unchanged Julia invocation and no array or follow-on submission.

Post-run reporting tests must also demonstrate that artifact locations and the approved diagnostic set are shown, missing artifacts are surfaced, and no scientific acceptance judgment is generated.

No numerical execution, CRC submission, dependency installation, or historical-file modification is part of those implementation tests.
