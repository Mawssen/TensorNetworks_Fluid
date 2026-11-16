#!/usr/bin/env python3
"""Synthetic-only cross-check of bounded-workspace derivatives and reductions."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import socket
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT / "legacy/mps_py_codes/calculate_spectral_derivative.py",
           ROOT / "legacy/mps_py_codes/calculate_u_comp_fidelity_l2norm.py",
           ROOT / "src/python/turbulence_tools_stats/reductions.py"]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def difference(a, b):
    import numpy as np
    delta = np.abs(a-b)
    mask = a != 0
    return {"bitwise_equal": bool(np.array_equal(a.view(np.uint64),b.view(np.uint64))),
            "max_absolute_difference": float(delta.max()),
            "max_relative_difference_nonzero": float((delta[mask]/np.abs(a[mask])).max()) if mask.any() else None,
            "zero_reference_mismatches": int(np.count_nonzero(b[~mask] != 0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--worker", choices=["derivative_reference","derivative_staged","metrics_reference","metrics_staged"])
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        parser.error("run inside a compute allocation")
    import numpy as np
    sys.path.insert(0,str(ROOT / "src/python"))
    from turbulence_tools_stats import spectral_derivative_slabbed, compare_fields_chunked
    derivative = load(SOURCES[0],"reference_derivative").spectral_derivative
    metrics = load(SOURCES[1],"reference_metrics")
    directory = args.output_directory.resolve()
    if args.worker:
        rng = np.random.default_rng(20261116)
        a = rng.standard_normal((128,128,128))
        b = a + 0.1*rng.standard_normal(a.shape)
        functions = {
            "derivative_reference": lambda: derivative(a,1,0.125),
            "derivative_staged": lambda: spectral_derivative_slabbed(a,1,0.125),
            "metrics_reference": lambda: np.array([metrics.calculate_fidelity(a,b),metrics.calculate_l2norm(a,b)]),
            "metrics_staged": lambda: np.array(list(compare_fields_chunked(a,b).values()))}
        calculate = functions[args.worker]
        calculate()
        timings = []
        for _ in range(3):
            start = time.perf_counter()
            result = calculate()
            timings.append(time.perf_counter()-start)
        np.save(directory/(args.worker+".npy"),result)
        (directory/(args.worker+".json")).write_text(json.dumps({"median_seconds":statistics.median(timings),
            "times_seconds":timings,"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2))
        return
    directory.mkdir(parents=True,exist_ok=False)
    hashes = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in SOURCES}
    for path in SOURCES[:2]:
        (directory/("reference-"+path.name)).write_bytes(path.read_bytes())
    rng = np.random.default_rng(91)
    report = {"job_id":os.environ["SLURM_JOB_ID"],"host":socket.gethostname(),
              "numpy":np.__version__,"source_sha256":hashes,"derivative_checks":[],
              "acceptance":"No scientific tolerance selected; differences reported, defaults unchanged",
              "benchmark":"128^3 synthetic Float64, median of 3 warm full calls; separate workers"}
    for shape in ((8,10,12),(7,9,11)):
        a = rng.standard_normal(shape).transpose(2,0,1)
        for axis in range(3):
            for order in (1,2):
                expected = derivative(a,axis,0.125,order)
                for slab in (1,3,20):
                    actual = spectral_derivative_slabbed(a,axis,0.125,order,slab_size=slab)
                    if actual.shape != a.shape or not np.isfinite(actual).all():
                        raise AssertionError("invalid derivative output")
                    report["derivative_checks"].append(dict(shape=list(a.shape),axis=axis,order=order,
                        slab_size=slab,**difference(expected,actual)))
    constant = np.ones((8,8,8),dtype=np.float64)
    for axis in range(3):
        np.testing.assert_array_equal(spectral_derivative_slabbed(constant,axis,0.125),np.zeros_like(constant))
    identical = compare_fields_chunked(constant,constant)
    assert identical == {"fidelity":1.0,"relative_l2":0.0}
    zero = compare_fields_chunked(np.zeros_like(constant),constant)
    assert np.isnan(zero["fidelity"]) and np.isinf(zero["relative_l2"])
    report["hand_checks"] = "constant derivatives, identical fields, zero-denominator behavior passed"
    for kind in ("derivative","metrics"):
        results = {}
        for version in ("reference","staged"):
            name = kind+"_"+version
            subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),
                "--output-directory",str(directory),"--worker",name],check=True,cwd=ROOT)
            results[version] = json.loads((directory/(name+".json")).read_text())
        a = np.load(directory/(kind+"_reference.npy"))
        b = np.load(directory/(kind+"_staged.npy"))
        results["differences"] = difference(a,b)
        results["speedup"] = results["reference"]["median_seconds"]/results["staged"]["median_seconds"]
        report[kind+"_benchmark"] = results
    for path in SOURCES:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[str(path.relative_to(ROOT))]
    report["sources_unchanged"] = True
    (directory/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({key:value for key,value in report.items() if key!="derivative_checks"},indent=2))


if __name__ == "__main__":
    main()
