#!/usr/bin/env python3
"""Compare preserved spectrum implementations on synthetic cubes on a compute node.

No production inputs, package installation, Slurm submission, or tolerance-based
scientific acceptance. Workers isolate each implementation's peak memory.
"""

import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import socket
import statistics
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "legacy/mps_py_codes/calculate_spectrum.py"
FLOAT32 = ROOT / "legacy/py/jhs_velocity_spectrum_stats.py"
OPTIMIZED = ROOT / "src/python/turbulence_tools_stats/spectrum.py"


def load_function(path):
    """Evaluate the literal historical function, excluding unrelated UI imports."""
    import numpy as np
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "calculate_spectrum"]
    if len(functions) != 1:
        raise RuntimeError("expected one historical calculate_spectrum in " + str(path))
    namespace = {"np": np, "gc": gc}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["calculate_spectrum"]


def worker(implementation, size, repetitions, directory):
    import numpy as np
    if implementation == "reference":
        calculate = load_function(directory / "reference.original.py")
    elif implementation == "legacy_float32":
        original = load_function(directory / "reference.float32.py")
        calculate = lambda *fields: original(*fields, verbose=False)
    else:
        sys.path.insert(0, str(ROOT / "src/python"))
        from turbulence_tools_stats import calculate_spectrum_binned
        calculate = calculate_spectrum_binned
    seed = 20261116 + size
    rng = np.random.default_rng(seed)
    fields = tuple(rng.standard_normal((size, size, size), dtype=np.float64) for _ in range(3))
    calculate(*fields)  # Same-size warmup; process/import startup excluded.
    timings = []
    for _ in range(repetitions):
        started = time.perf_counter()
        k, spectrum = calculate(*fields)
        timings.append(time.perf_counter() - started)
    result = {"implementation": implementation, "size": size, "seed": seed,
              "k": k.tolist(), "spectrum": spectrum.tolist(),
              "spectrum_dtype": str(spectrum.dtype), "times_seconds": timings,
              "median_seconds": statistics.median(timings),
              "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    with (directory / (implementation + "_" + str(size) + ".json")).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


def differences(reference, candidate):
    import numpy as np
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    absolute = np.abs(reference - candidate)
    nonzero = reference != 0
    return {"bitwise_equal": bool(np.array_equal(reference.view(np.uint64), candidate.view(np.uint64))),
            "max_absolute_difference": float(absolute.max()),
            "max_relative_difference_on_nonzero_reference":
                float((absolute[nonzero] / np.abs(reference[nonzero])).max()) if nonzero.any() else None,
            "zero_reference_mismatches": int(np.count_nonzero(candidate[~nonzero] != 0)),
            "acceptance": "not evaluated; no scientific tolerance has been selected"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--worker", choices=["reference", "binned", "legacy_float32"])
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        parser.error("numerical comparisons require an existing compute allocation")
    if len(set(args.sizes)) != len(args.sizes) or any(size < 2 for size in args.sizes) or args.repetitions < 1:
        parser.error("use distinct sizes >= 2 and positive repetitions")
    directory = args.output_directory.resolve()
    if args.worker:
        if len(args.sizes) != 1:
            parser.error("a worker requires exactly one size")
        worker(args.worker, args.sizes[0], args.repetitions, directory)
        return 0

    import numpy as np
    directory.mkdir(parents=True, exist_ok=False)
    sources = {"reference": ORIGINAL.read_bytes(), "legacy_float32": FLOAT32.read_bytes(),
               "binned": OPTIMIZED.read_bytes()}
    for name, filename in (("reference", "reference.original.py"),
                           ("legacy_float32", "reference.float32.py")):
        (directory / filename).write_bytes(sources[name])
    report = {"host": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "python": sys.version, "python_executable": sys.executable,
              "numpy": np.__version__, "platform": platform.platform(),
              "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()},
              "scope": "synthetic data only; historical shell/Float64 convention is the reference",
              "timing": "median of warmed full-function calls, including FFTs and binning",
              "memory": "separate-worker Linux process peak RSS, including inputs and warmup",
              "thread_environment": {key: os.environ.get(key) for key in
                  ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
              "cases": []}
    suite = unittest.defaultTestLoader.discover(str(ROOT / "test/python"), pattern="test_spectrum_binning.py")
    with (directory / "unit_tests.log").open("x") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report["unit_tests"] = {"successful": result.wasSuccessful(), "run": result.testsRun,
                            "skipped": len(result.skipped)}
    if not result.wasSuccessful() or result.skipped:
        (directory / "report.json").write_text(json.dumps(report, indent=2))
        raise RuntimeError("spectrum unit checks failed or skipped; inspect unit_tests.log")
    for size in args.sizes:
        results = {}
        for implementation in ("reference", "binned", "legacy_float32"):
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                "--output-directory", str(directory), "--sizes", str(size),
                "--repetitions", str(args.repetitions), "--worker", implementation], check=True, cwd=ROOT)
            results[implementation] = json.loads((directory / (implementation + "_" + str(size) + ".json")).read_text())
        reference, binned = results["reference"], results["binned"]
        for implementation in ("binned", "legacy_float32"):
            if results[implementation]["k"] != reference["k"]:
                raise RuntimeError("wavenumber definitions differ")
        case = {"size": size, "reference_seconds": reference["median_seconds"],
                "binned_seconds": binned["median_seconds"],
                "legacy_float32_seconds": results["legacy_float32"]["median_seconds"],
                "speedup_reference_over_binned": reference["median_seconds"] / binned["median_seconds"],
                "reference_peak_rss_kib": reference["process_peak_rss_kib"],
                "binned_peak_rss_kib": binned["process_peak_rss_kib"],
                "binned_vs_reference": differences(reference["spectrum"], binned["spectrum"]),
                "legacy_float32_vs_reference": differences(reference["spectrum"], results["legacy_float32"]["spectrum"])}
        report["cases"].append(case)
        np.savez(directory / ("spectra_" + str(size) + ".npz"), k=reference["k"],
                 reference=reference["spectrum"], binned=binned["spectrum"],
                 legacy_float32=results["legacy_float32"]["spectrum"])
        (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(case), flush=True)
    for name, path in (("reference", ORIGINAL), ("legacy_float32", FLOAT32), ("binned", OPTIMIZED)):
        if path.read_bytes() != sources[name]:
            raise RuntimeError("source changed during comparison: " + str(path))
    report["source_unchanged"] = True
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Comparison report: " + str(directory / "report.json"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
