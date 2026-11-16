#!/usr/bin/env python3
"""Copyable synthetic statistics case using the preferred implementations."""
from pathlib import Path
import json
import os
import socket
import sys

# EDIT CASE VALUES HERE. Replace the synthetic loader after inputs are approved.
shape = (8, 8, 8)
output_directory = Path("artifacts/runs/statistics/synthetic-example")

if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
    raise SystemExit("run inside a compute allocation")
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/python"))
from turbulence_tools_stats import calculate_spectrum, compare_fields

u = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
v, w = np.zeros_like(u), np.zeros_like(u)
candidate_u = 0.99 * u
k, spectrum = calculate_spectrum(u, v, w)       # Preferred Float64 binning.
metrics = compare_fields(u, candidate_u)       # Preferred chunked reductions.
# To reproduce the original implementations, use implementation="reference".
output_directory = ROOT / output_directory
output_directory.mkdir(parents=True, exist_ok=False)
np.savez(output_directory / "spectrum.npz", k=k, spectrum=spectrum)
(output_directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(output_directory)
