# Generated artifacts (not committed)

Use `artifacts/runs/<method>/<dataset-field-timestep>/<shape>/<ordering>/<mode>/<run-id>/`.
Each invocation gets a new directory containing its log, input/parameter record,
execution state, and reconstructed output. Never overwrite an earlier run.

For the prepared JHS case:

```text
artifacts/runs/mps/jhs_u_t1/1024cubed/interleaved/cutoff_1e-4/<run-id>/
  case.toml
  run.log
  truncated_U_t1_1024_il_cf0.0001.mat
```

Only this README is tracked under artifacts. Synthetic examples and CRC comparisons
from the package's initial validation remain under ignored `codex_transcripts/`.
