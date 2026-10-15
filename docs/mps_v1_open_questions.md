# MPS v1 open-question register

## First-interface status — 2026-10-01

The owner has authorized the first declaration-only [dry-run interface](mps_v1_dry_run_interface.md). Its standard-library JSON schema is a routine engineering choice for the available Python 3.9, which lacks standard-library TOML parsing. The [JSON template](mps_v1_dry_run_template.json) is the accepted template for this stage; the commented TOML inventory remains background for the broader design.

For Q04, the owner chose temporary numeric-cutoff rejection with `chi=N` support. Q04 itself remains unresolved; no formatting table or cutoff formatter is approved. This does not change the numerical authority or its cutoff workflow. Q02 is limited to validating source-established codes and literal filename tags; semantic bit interpretation remains open. Q03 checks only integer-token syntax, not scientific range or target Julia representability. Other unanswered policy items below remain future complete-preflight/submission gates and are printed as unverified by this first interface. No data/result access, scheduler operation, or real execution is authorized. Earlier documentation-only notices below describe the prior milestone.

This is the single current register for the unchanged-historical-CLI automation milestone. No option below is selected. The owner is unavailable; record blockers without asking questions, implementing code, inspecting datasets/results/environments, or contacting Slurm. Approved decisions are in the [automation plan](mps_automation_plan.md#status-and-boundary); the [template](mps_v1_config_template.toml) intentionally cannot describe a runnable case until completed. The [dry-run spec](mps_v1_dry_run_spec.md) references these IDs.

Stage labels: **D** blocks a complete dry-run plan; **R** blocks real submission or its required post-run handoff; **S** concerns scientific validation. “Before code” distinguishes unresolved validator/compatibility design from explicit per-case values that may remain unset while a generic interface is designed. No code is authorized now, regardless of stage. A blocked plan reports all missing decisions and cannot become submission-ready.

## Q01 — Input file length

- Question: Require exactly 2,006,581,248 bytes, or permit additional trailing bytes?
- Evidence: [Julia/0mps_calc.jl:825–836](../Julia/0mps_calc.jl#L825-L836) reads only `4*864*1008*576` bytes and reshapes them; it does not check EOF.
- Options/consequence: exact size detects unexpected headers/trailers; at-least size preserves the historical ignored-tail behavior but cannot rule out a layout mismatch. Short files cannot fill the required grid under either option.
- Gate: **D, R**. Before code: choose the file-size validation policy. No data scan is authorized now.

## Q02 — Ordering contract and per-case selection

- Question: Which code will each case use, and what exact bit/site naming should the configuration label assert?
- Evidence: [ordering and inverse, lines 317–361](../Julia/0mps_calc.jl#L317-L361), [permutations, lines 181–315](../Julia/0mps_calc.jl#L181-L315), and [output tags, lines 852–856](../Julia/0mps_calc.jl#L852-L856). Helpers include 5/6 but the active output path supports only 1–4. Code 1 in the [Slurm array, lines 29–31](../Julia/00mps_calc.slurm#L29-L31) is not a default.
- Options/consequence: expose codes 1–4 with literal historical tags and no extra bit interpretation; or require an owner-approved semantic bit/site map as well. Either invokes the same historical functions. Each later case must independently select one supported code.
- Gate: **D, R**; semantic interpretation also **S**. Before code: settle label validation; case code stays deferred. No permutation is reimplemented.

## Q03 — Truncation domain and per-case value

- Question: Which numeric inputs should v1 accept, and which single mode/value will a case use?
- Evidence: [parsing, lines 811–819](../Julia/0mps_calc.jl#L811-L819), [MPS branches, lines 381–394](../Julia/0mps_calc.jl#L381-L394), and [active call, lines 846–850](../Julia/0mps_calc.jl#L846-L850). Julia parses a Float64 cutoff or integer cap without wrapper-style range validation; zero cutoff occurs in the source.
- Options/consequence: require finite positive cutoffs and positive integer caps; or additionally permit cutoff zero with its explicitly approved computational scope. A more literal parser-compatible domain would require the owner to specify treatment of negative/non-finite values and invalid caps. These choices change what the wrapper will reject, not the historical implementation.
- Gate: **D, R**. Before code: numeric-domain policy. Mode and value remain explicit later case decisions. Earlier positive-only planning prose was a proposal, not authorization.

## Q04 — Exact Julia filename formatting

- Question: How will a planner that does not run Julia reproduce cutoff output labels exactly?
- Evidence: [Float64 parse, line 818](../Julia/0mps_calc.jl#L818) and [interpolated labels, lines 858–865](../Julia/0mps_calc.jl#L858-L865). Input spelling is not necessarily output spelling.
- Options/consequence: approve a verified Julia-compatible formatting implementation for an agreed runtime and token domain; or approve a bounded table of exact token/label pairs established independently for that runtime. The latter limits supported cutoffs. A user-entered unverified filename alone is insufficient. Running Julia during dry-run contradicts the current design and would need a separate scope change.
- Gate: **D, R** for cutoff cases. Before code: compatibility mechanism and permitted token domain. Unknown labels must block, never approximate.

## Q05 — Case identity and unit approval

- Question: What dataset identifier, field, timestep, absolute input path, and nonempty unit string will each future case declare?
- Evidence: [CLI arguments, lines 807–809](../Julia/0mps_calc.jl#L807-L809), [basename extraction, lines 821–824](../Julia/0mps_calc.jl#L821-L824), and conflicting [Slurm prose/arrays, lines 13–31](../Julia/00mps_calc.slurm#L13-L31).
- Options/consequence: owner supplies one complete case and approves its field/unit pairing; or leaves it unselected and only an incomplete plan is possible. No roster, discovery, or unit registry is an option within approved v1.
- Gate: **D, R** per case. Before code: no production selection required; mandatory-field behavior is already decided.

## Q06 — Output root, logs, and audit destinations

- Question: Which absolute working directory and new audit/log destinations will a case use?
- Evidence: [relative output directory, lines 821–824](../Julia/0mps_calc.jl#L821-L824), [MAT write mode, lines 416–425](../Julia/0mps_calc.jl#L416-L425), and [historical log pattern, Slurm line 3](../Julia/00mps_calc.slurm#L3).
- Options/consequence: a new owner-designated case directory isolates output; or an approved existing working directory preserves an established location, provided the exact target is absent. Separate or combined stdout/stderr may be explicitly chosen. Audit artifacts require a separate non-colliding destination. Neither choice permits overwriting results.
- Gate: **D, R** per case. Before code: directory-creation and log-template policy must be specified; actual paths can wait.

## Q07 — Julia and package environment

- Question: Which executable, project/depot, package versions, startup behavior, modules, and BLAS environment are authoritative?
- Evidence: [imports and MKL, lines 1–13](../Julia/0mps_calc.jl#L1-L13); [Slurm executable, lines 18–19](../Julia/00mps_calc.slurm#L18-L19) supplies a path but no package-version contract.
- Options/consequence: approve the existing captured environment as-is; or nominate another already available environment and approve its compatibility explicitly. Inheritance of startup/project/depot versus explicit selection must be recorded. Neither option authorizes installation or upgrades.
- Gate: **D** for the exact execution command/setup, **R**, and historical comparisons **S**. Before code: environment representation/capture policy; concrete environment can remain deferred.

## Q08 — Thread and process allocation

- Question: Which Julia/OMP settings and CPU/task allocation will accompany the unchanged BLAS count of 10?
- Evidence: [BLAS setting, line 12](../Julia/0mps_calc.jl#L12), [Slurm CPUs/tasks, lines 4–6](../Julia/00mps_calc.slurm#L4-L6), [OMP setting, line 18](../Julia/00mps_calc.slurm#L18).
- Options/consequence: approve an explicit environment/allocation consistent with the fixed BLAS count; or provide another explicit setting and acknowledge its scheduling/oversubscription implications. Do not infer Julia thread count from OMP or CPUs. One invocation must not become multiple replicas under a launcher.
- Gate: **D, R**. Before code: permitted task/launcher model and consistency checks; actual CPU and thread choices remain per-case.

## Q09 — CRC destination and resources

- Question: Which cluster, partition, account, QOS, reservation/constraints, node/task/CPU counts, memory, walltime, and job name are approved?
- Evidence: [historical request, Slurm lines 2–11](../Julia/00mps_calc.slurm#L2-L11). It is a 28-task campaign request, not a validated single-case resource estimate.
- Options/consequence: explicitly approve selected historical values for a single case; or provide another explicit resource request. Optional scheduler fields must be explicitly unset when unused. Neither path inherits the array or its concurrency.
- Gate: **D, R** per case. Before code: resource representation and validation policy; no allocation is selected here and no scheduler query is permitted now.

## Q10 — Runtime and memory estimate evidence

- Question: Is there an owner-approved comparable retained accounting record for the case?
- Evidence: [historical walltime/memory requests, Slurm lines 9–10](../Julia/00mps_calc.slurm#L9-L10) and [Julia elapsed print, line 870](../Julia/0mps_calc.jl#L870) provide no observed run measurement in those sources.
- Options/consequence: supply comparable retained measurements and their provenance; or retain explicit unknown runtime/peak memory. Requested limits remain distinct from estimates. The size-derived execution scope is still printable.
- Gate: not **D** (unknown is permitted); **R** requires owner review of the request and known uncertainty. Before code: no benchmark required; no profiling run is authorized.

## Q11 — Input identity and provenance cost

- Question: How should input identity be bound to approval without an unapproved full-data read?
- Evidence: [input read, lines 831–836](../Julia/0mps_calc.jl#L831-L836) consumes about 1.869 GiB; [the plan's provenance requirement](mps_automation_plan.md#mandatory-dry-run-report) requires input and configuration tracking.
- Options/consequence: record path/size/mtime and a supplied immutable identifier/checksum with its provenance, which is cheaper but may not detect same-size mutation; or approve a full hash pass with its I/O scope displayed. Symlink target identity must be explicit, not silently used to change the historical basename.
- Gate: **D, R**. Before code: identity and revalidation policy. No dataset hash is computed in this milestone.

## Q12 — Source pinning and dirty trees

- Question: Which source revision will run, and how should dirty source/runtime changes affect eligibility?
- Evidence: [authoritative active path, lines 806–870](../Julia/0mps_calc.jl#L806-L870) and [plan provenance](mps_automation_plan.md#mandatory-dry-run-report) require commit plus script hash, not a commit string alone.
- Options/consequence: require a clean pinned calculation/planner revision; or permit explicitly approved dirty state with captured file hashes/diff provenance. A commit alone cannot identify dirty content. Either policy must detect changes before launch and invalidate approval.
- Gate: **D, R**. Before code: dirty-state policy and source-change handling. Revision selection is a later case/release choice.

## Q13 — Approval record and lifetime

- Question: What durable record identifies the owner, binds the exact plan digest, and defines expiry/revalidation?
- Evidence: [manual submission gate](mps_automation_plan.md#approval-and-manual-submission-gate); the [historical Slurm invocation, lines 50–51](../Julia/00mps_calc.slurm#L50-L51) has no approval mechanism.
- Options/consequence: explicit owner attestation tied to an immutable plan; or an authenticated/signed record. Choose an explicit lifetime or no time expiry with mandatory change invalidation. All preserve manual submission and owner-only scientific acceptance.
- Gate: **R**, not basic unapproved **D**. Before code: approval representation for any approval/verification component; no approval is implied by a template value.

## Q14 — Collision and queued-job revalidation

- Question: How should future execution prevent an output or source change between review, submission, and job start?
- Evidence: [MAT opens in write mode, line 418](../Julia/0mps_calc.jl#L418) and [relative output creation, lines 823–824](../Julia/0mps_calc.jl#L823-L824).
- Options/consequence: exclusive new case directories plus launch-time source/input/output checks; or an approved shared-directory locking/reservation design plus those checks. A planning-time existence check alone leaves a race. Neither permits overwriting or patching the historical writer.
- Gate: **R**; affects prepared launch script **D**. Before code: filesystem/concurrency strategy. A failed/partial result is reported, not deleted or retried automatically.

## Q15 — Structural diagnostics collection and artifacts

- Question: Which existing reader, execution context, collection permissions, and report format will supply the approved structural diagnostics?
- Evidence: [MAT variables, lines 416–425](../Julia/0mps_calc.jl#L416-L425), [`chi_crit=linkdims`, line 398](../Julia/0mps_calc.jl#L398), [timing, lines 847–870](../Julia/0mps_calc.jl#L847-L870), and [approved handoff](mps_automation_plan.md#mandatory-post-run-review-handoff).
- Options/consequence: collect metadata and the full `chi_crit` sequence in the already approved compute context using an approved available reader; or perform a separately authorized read-only post-run collection. Accounting may come from retained records or later-authorized scheduler queries. Select report/artifact format and paths explicitly. Missing observations stay unavailable. Reader dimension conventions must be labeled; do not transpose `u` or load it solely to calculate statistics.
- Gate: **R** handoff; collection design/scope affects **D** if placed in the job. Before code: reader/environment, metadata orientation, artifact format and collection mechanism. No Slurm query, result read, package install, or scientific metric is authorized now.

## Q16 — Schema, parser, and preparation interface

- Question: Which already available implementation runtime/TOML parser, schema version, command names, optional-field representation, and plan-artifact format should be frozen?
- Evidence: [historical CLI, lines 807–819](../Julia/0mps_calc.jl#L807-L819) has three arguments and no configuration parser; the [commented template](mps_v1_config_template.toml) is a documentary inventory, not an implemented schema.
- Options/consequence: a standard-library TOML-capable runtime with a small preparation interface; or another owner-approved existing runtime/parser with explicit compatibility requirements. Choose explicit unset representation and exact raw-token preservation. No dependency install is implicit. Original bytes and canonical configuration must be separately recorded; define deterministic digest coverage excluding the digest itself and subsequent approval record.
- Gate: **D**. Before code: concrete interface/parser and serialization choices. Routine engineering choices can later be resolved within authorization, but are not invented in this documentation pass.

## Q17 — Filesystem and remote preflight policy

- Question: Where will preparation run, and how will missing CRC-visible paths, symlinks, and write permissions be validated?
- Evidence: [relative historical invocation and input, Slurm lines 48–51](../Julia/00mps_calc.slurm#L48-L51) and [cwd-relative output, Julia lines 821–824](../Julia/0mps_calc.jl#L821-L824) depend on execution context.
- Options/consequence: prepare on the approved CRC filesystem with read-only path checks; or prepare elsewhere and mark checks unverified until an authorized CRC-side verification. Reject symlinks, or explicitly record both requested path/basename and target identity. Define audit-directory creation permissions without probing historical outputs by writing to them.
- Gate: **D** for a fully verified plan, **R**. Before code: verification context/path policy. Offline reports cannot claim verified file availability or scheduler eligibility.

## Q18 — Structural acceptance fixtures

- Question: What non-numerical fixtures establish argument, label, template, and diagnostic-reader compatibility?
- Evidence: [historical output naming, lines 852–866](../Julia/0mps_calc.jl#L852-L866), [MAT writer, lines 416–425](../Julia/0mps_calc.jl#L416-L425), and the older [baseline plan](../validation/mps_v1_baseline_plan.md#phase-1-synthetic-rank-one-test) concern different testing scopes.
- Options/consequence: owner-approved static text/metadata fixtures and known formatting examples; or separately authorized existing MAT metadata fixtures. Do not select a historical result, generate a numerical tensor baseline, or interpret a numerical mismatch as a historical bug. Metadata fixtures alone do not validate reconstructed field accuracy.
- Gate: implementation acceptance for **D/R**; does not establish **S**. Before code: compatibility fixtures/expected metadata conventions; no historical run is needed for text tests.

## Q19 — Scientific validation beyond v1

- Question: If a later milestone requires numerical validation, which reference outputs, formulas, normalization, tolerances, and discrepancy policy will the owner approve?
- Evidence: [active calculation, lines 366–413](../Julia/0mps_calc.jl#L366-L413) defines no fidelity metric; the older [baseline acceptance criteria](../validation/mps_v1_baseline_plan.md#acceptance-criteria) leave numerical tolerances unresolved.
- Options/consequence: continue owner review using v1 structural observations; or explicitly authorize a separate validation milestone with named references and definitions. No acceptance threshold, normalization change, filtering operator, or bug correction is implied.
- Gate: **S only**, outside v1 and not a prerequisite to writing a generic planning wrapper. Do not reopen approved structural-only scope.

## Q20 — Readiness decisions for the other three candidates

- Question: What scientific and compatibility contracts must be frozen for DNS statistics, CRC block filtering, and cross-source comparison before their implementation?
- Evidence: [DNS driver, lines 54–94](../py/run_dns_stats.py#L54-L94), [block-filter runner, lines 1–39](../LES_interface/crc/run_les.jl#L1-L39), [filter implementation, line 3 onward](../LES_interface/LES_filter.jl#L3), [comparison discovery, lines 148–264](../py/plot_dns_mps_comparison.py#L148-L264), and [optional recomputation, lines 1051–1083](../py/plot_dns_mps_comparison.py#L1051-L1083).
- Options/consequence: for DNS, approve named inputs/region/formulas/operators/statistics and environment or leave standardization blocked; for filtering, approve the non-overlapping block operator, input layout/factor/invariants and packaging plan or retain historical status; for comparison, approve exact existing source artifacts, semantic compatibility and plotting-only behavior, or separately scope optional recomputation. Each also needs outputs, acceptance policy and any CRC request approved. No selection transfers from MPS automatically.
- Gate: **D/R/S** for those workflows, not MPS implementation. Before their code: freeze the contracts detailed in the [catalog readiness notes](workflow_catalog.md#implementation-readiness-notes-for-the-top-three).

## Reconciled earlier decision lists

The [earlier decisions](mps_v1_decisions.md), [reference contract](mps_v1_reference_contract.md), and [baseline plan](../validation/mps_v1_baseline_plan.md) predate this wrapper-only milestone. Their numerical-rebuild recommendations do not select ordering 1, require a synthetic compression run, or authorize fidelity calculations in v1. Numerical authority, fixed shape/dtype/endian/crop/axes, mandatory units with owner approval, case-agnostic selection, structural diagnostic scope, and owner-only scientific acceptance are resolved. Arbitrary-shape handling and scientific changes are outside this fixed-CLI scope. The Slurm prose/array mismatch is preserved as evidence; deciding which was intended is unnecessary because neither defines the future roster.
