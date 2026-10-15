# First executable MPS v1 dry-run interface

This interface validates explicit declarations and prints a proposal. It is the first bounded implementation of the [automation plan](mps_automation_plan.md), not the complete [CRC preflight specification](mps_v1_dry_run_spec.md). The owner authorized this stage on 2026-10-01. The historical calculation remains the opaque active three-argument CLI in `Julia/0mps_calc.jl`.

## Usage and configuration

Use the single [JSON template](mps_v1_dry_run_template.json), replacing every `REQUIRES_OWNER_DECISION` value with an explicitly selected case declaration. The template intentionally fails validation until completed. No production field, dataset, timestep, path, ordering, cap, output root, or resource request is supplied by the template.

```sh
python3 -B scripts/mps_v1_dry_run.py /absolute/path/to/one-case.json
```

Python 3.9 or newer and its standard library suffice; nothing is installed. JSON was chosen as a routine engineering decision because the available Python 3.9 lacks a standard-library TOML parser. The original [commented TOML inventory](mps_v1_config_template.toml) remains documentation for a broader future schema and is not accepted by this command.

The JSON schema has required sections `case`, `historical`, `runtime`, `slurm`, `audit`, `approval`, and `diagnostics`, plus integer `schema_version=1`. All keys shown in the template are mandatory; unknown keys, duplicates, control characters, unpaired Unicode surrogates and placeholders are errors. Non-finite JSON numbers, including overflowing exponent notation, are rejected. Malformed or excessively nested JSON fails with the blocked status, not a traceback. No defaults are filled. Fixed axis/shape/bounds tuples are permitted; selector lists and scientific processing options are rejected.

Every case must explicitly declare dataset identifier, field, timestep string, absolute input path, nonempty unit string, and the approved Jet Flame DNS contract: big-endian Float32; `(x,y,z)=(864,1008,576)` with x fastest; centered `(512,512,512)` crop before tensorization; zero-based half-open bounds `x=[176,688)`, `y=[248,760)`, `z=[32,544)`. Timesteps must exactly match the input basename suffix, preserving leading zeros. Units remain owner-approved metadata; no field registry is created.

Select one ordering code 1–4 and its literal historical tag (`il`, `seq`, `comb1`, `combn`). This validates the source's filename map without selecting or interpreting bit significance. The authority path is fixed relative to the planner's repository; alternative historical scripts cannot be configured.

## Temporary chi-only limitation

The owner explicitly approved supporting `chi=N` while rejecting numeric cutoffs until **Q04** is resolved. Use `case.truncation.kind="max_bond_dimension"` and an integer token string for `value`. The command passes that token as `chi=<token>` and derives the filename label from its integer value. No cutoff-label table or float formatter is invented. This is an interface limitation, not a change to the historical numerical workflow.

The first stage checks integer-token syntax only. It deliberately does not settle Q03's scientific numeric-domain policy or certify the target runtime's Julia `Int` representability. Thus even a syntactically valid zero, negative, or very large cap is a proposal requiring further review, not an assertion that Julia will accept it or the scientific case is appropriate.

## Output and safety boundary

Exit status 0 means **declarations valid**, not ready to submit. Standard output is a JSON report containing the exact argv list and shell-quoted historical command, explicit working directory/environment declarations, unchanged historical filename, proposed absolute MAT path, and proposed provenance paths. The report also includes the requested resources verbatim, structural diagnostic scope, owner identity, size-derived execution scope, unknown runtime/memory estimates, Git commit/dirty state, source/planner/configuration hashes, canonical configuration, and a deterministic proposal digest excluding itself.

Exit status 2 means blocked configuration or unavailable local provenance; reasons appear on stderr. Neither result creates files or directories. Proposed provenance paths are `config.original.json`, `config.canonical.json`, `plan.json` and `plan.txt` in the declared audit directory. They are not written. stdout/stderr paths must be distinct in this initial schema; a future approved joined-log policy may extend it. The audit directory must be separate from the derived result directory. Lexical checks reject equality or file-as-parent conflicts among input, output, logs and proposed provenance files, and declared files conflicting with the working/result/audit directories. These checks do not establish filesystem identity, permissions, symlink targets, absence of existing files or race safety.

The required future process cwd is the explicit `historical.working_directory`, not the directory containing the Julia script or the planner's current directory. Julia creates the relative `truncated_<timestep>` directory there. The printed payload does not change directories or apply the separately reported environment; executing it elsewhere would change the output location. The argument order is always `<absolute Julia executable> <fixed authority> <absolute input> chi=<token> <split>` ([active CLI](../Julia/0mps_calc.jl#L806), [relative output directory](../Julia/0mps_calc.jl#L821), [tag/filename construction](../Julia/0mps_calc.jl#L853)).

The command opens only its supplied configuration, its own source, and the approved Julia source for provenance. It runs only read-only Git metadata commands. It never opens, stats, resolves, or hashes the DNS path; never checks runtime executables, packages, remote permissions, or scheduler eligibility; never creates Slurm scripts, invokes Julia or `sbatch`, reconstructs data, or computes diagnostics/fidelity/error metrics. The printed Julia command is a proposed payload with separately displayed cwd, environment and module declarations, not an executable job plan. Environment setup is not evaluated or certified.

Filesystem checks, input-length policy, input identity, approved dirty-state handling, cap domain, resource/thread/launcher validation, output collision/race protection, runtime verification, approval records, and diagnostic collection remain explicitly unverified under [Q01–Q18](mps_v1_open_questions.md). Resource values are declarations only; this stage verifies their presence/types, not Slurm-specific eligibility or request grammar. No number is inherited from the historical array. The existing broader spec still governs later CRC preparation. A complete CRC plan must fail closed on these gates before real submission; this declaration proposal cannot satisfy them by itself.

## Safe checks

```sh
python3 -B -m unittest discover -s tests -p test_mps_v1_dry_run.py -v
git diff --check
```

Tests use invented text metadata and temporary JSON files. They cover missing declarations, all fixed input-contract mismatches, invalid scalar types, leading-zero timestep mismatch, unsupported ordering/modes, Q04 cutoff rejection, unknown processing/array/source-override keys, duplicate JSON keys, non-finite JSON constants/exponents, excessive nesting, placeholders, controls, invalid Unicode, provenance/file/directory collisions, integer filename normalization, exact argv order, cwd-independent output derivation, crop bounds from the historical centering formula, quoting, and absence of configured DNS-path access or payload execution through the CLI. They do not create DNS fixtures or validate numerical results. Q03's cap-domain and target-`Int` checks remain unverified, including for syntactically valid zero, negative and oversized tokens; the tests preserve that explicit boundary rather than select a policy.
