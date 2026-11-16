#!/usr/bin/env python3
"""Execute one explicitly selected chi case within an existing compute allocation."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from mps_v1_dry_run import build_report, parse_configuration, ConfigurationError


def execute(config, raw):
    # Share validation, argv, naming, and provenance with the unchanged dry run.
    plan = build_report(config, raw)
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().split(".")[0].startswith("login"):
        raise ConfigurationError("execution requires an existing CRC compute allocation")
    cwd = Path(plan["working_directory"])
    if not cwd.is_dir():
        raise ConfigurationError("working directory must already exist: " + str(cwd))
    output = Path(plan["proposed_output_path"])
    if os.path.lexists(output):
        raise ConfigurationError("output already exists: " + str(output))
    if not Path(config["case"]["input_path"]).is_file():
        raise ConfigurationError("input must be an existing file")
    if not os.access(plan["argv"][0], os.X_OK):
        raise ConfigurationError("Julia executable is unavailable")
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = plan["environment_declarations"]["OMP_NUM_THREADS"]
    threads = plan["environment_declarations"]["JULIA_NUM_THREADS"]
    if threads == "unset":
        environment.pop("JULIA_NUM_THREADS", None)
    else:
        environment["JULIA_NUM_THREADS"] = threads
    logs = [Path(config["slurm"][key]) for key in ("stdout_path", "stderr_path")]
    for path in logs:
        if os.path.lexists(path):
            raise ConfigurationError("log already exists: " + str(path))
    audit = Path(config["audit"]["directory"])
    for path in [Path(config["case"]["input_path"]), *logs]:
        if audit == path or audit in path.parents:
            raise ConfigurationError("input and logs must be outside the audit directory")
    audit.mkdir(parents=True, exist_ok=False)
    paths = plan["proposed_provenance_paths"]
    Path(paths["original_config"]).write_bytes(raw)
    Path(paths["canonical_config"]).write_text(plan["provenance"]["canonical_config"] + "\n")
    Path(paths["plan"]).write_text(json.dumps(plan, indent=2) + "\n")
    Path(paths["summary"]).write_text(plan["historical_command"] + "\n")
    result = {"state": "STARTING", "argv": plan["argv"], "cwd": str(cwd),
              "output": str(output), "logs": [str(p) for p in logs],
              "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "module_environment_declaration": config["runtime"]["module_environment"],
              "environment_setup": "inherited; module declaration recorded, never evaluated"}
    started = time.monotonic()
    status = 1
    try:
        for path in logs:
            path.parent.mkdir(parents=True, exist_ok=True)
        with logs[0].open("x") as stdout, logs[1].open("x") as stderr:
            child = subprocess.run(plan["argv"], cwd=cwd, env=environment,
                                   stdout=stdout, stderr=stderr, check=False)
        result["returncode"] = child.returncode
        status = child.returncode if child.returncode > 0 else (128 - child.returncode if child.returncode < 0 else 0)
        if status == 0 and not output.is_file():
            status = 1
            result["error"] = "Julia exited successfully but expected MAT output is missing"
        result["state"] = "SUCCEEDED" if status == 0 else "FAILED"
    except (OSError, KeyboardInterrupt) as error:
        result.update(state="FAILED", error=str(error))
        status = 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["output_exists"] = output.is_file()
        result["output_bytes"] = output.stat().st_size if output.is_file() else None
        result["MAT_diagnostics"] = "unavailable; MAT contents not inspected"
        (audit / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return status, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration")
    args = parser.parse_args(argv)
    try:
        raw = Path(args.configuration).read_bytes()
        status, result = execute(parse_configuration(raw.decode("utf-8")), raw)
    except (ValueError, RecursionError, UnicodeError, OSError, subprocess.SubprocessError) as error:
        print("RUN BLOCKED: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return status


if __name__ == "__main__":
    sys.exit(main())
