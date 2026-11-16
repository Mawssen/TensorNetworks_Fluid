# Repository-wide Codex instructions

This file applies to the entire repository and to every future Codex session working in it.

## New MPS and LES cases

Start from the closest script in `cases/` (templates in `cases/examples/`), keep editable case values together
near its top, and reuse the public functions via `using TurbulenceTools` with
the repository project (`julia --project=.`). Ask only for missing scientific
inputs that affect the result; reuse inputs already approved for the case.
Keep the LES operator explicit and dataset/CRC settings outside the package.
Keep generated results and logs in a new parameter-specific directory under
ignored `artifacts/runs/`; keep reusable code, reference indexes, and tests separate.
The existing approval requirements for dependency changes, production runs and
CRC job submission still apply.

For new Python statistics cases, use the preferred `turbulence_tools_stats`
entry points from `src/python`: `calculate_spectrum` uses Float64 binning and
`compare_fields` uses chunked fidelity/L2 by default (owner-approved after the
1024³ JHS cross-check). Keep `spectral_derivative` on its original CPU default;
slabbed derivatives are explicit memory-saving alternatives. Preserve the
`implementation="reference"` fallback and the historical sources in `legacy/`.

## Evidence and decision policy

Before making any scientific, compatibility, or costly-computing decision that is not directly established by repository evidence, stop and ask the user one concise question. State the decision, the relevant evidence, the available options, and the consequence of each option. Do not silently choose among conflicting historical scripts, formulas, or conventions.

Ask the user before:

- selecting an authoritative historical script when multiple scripts differ;
- choosing tensor index ordering, bit significance, crop, normalization, truncation, fidelity definition, filtering operator, physical formula, units, or axis convention;
- interpreting a difference as a historical bug rather than intended behavior, or vice versa;
- selecting datasets, timesteps, fields, or regression outputs;
- changing, moving, deleting, or rewriting historical code;
- installing or upgrading dependencies;
- submitting, cancelling, or materially increasing CRC jobs or resources.

For routine engineering decisions that do not affect scientific meaning, proceed and document the decision.

When a question is required, keep it concise while including:

1. the decision that is blocked;
2. the repository evidence relevant to it;
3. the available options; and
4. the consequence of each option.
