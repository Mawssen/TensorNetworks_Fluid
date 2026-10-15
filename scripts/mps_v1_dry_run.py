#!/usr/bin/env python3
"""Validate declarations and print an MPS proposal; never execute its payload."""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parent.parent
AUTHORITY = REPOSITORY / "Julia" / "0mps_calc.jl"
TAGS = {1: "il", 2: "seq", 3: "comb1", 4: "combn"}
PROVENANCE_FILES = {
    "original_config": "config.original.json",
    "canonical_config": "config.canonical.json",
    "plan": "plan.json",
    "summary": "plan.txt",
}
FIXED_INPUT = {
    "domain": "Jet Flame DNS",
    "scalar_type": "Float32",
    "endianness": "big-endian",
    "full_grid_shape": [864, 1008, 576],
    "axis_order": ["x", "y", "z"],
    "fastest_axis": "x",
    "crop_kind": "centered",
    "crop_shape": [512, 512, 512],
    "crop_indexing": "zero-based-half-open",
}


class ConfigurationError(ValueError):
    pass


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigurationError("duplicate key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ConfigurationError("non-finite JSON number: " + value)


def parse_configuration(text):
    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ConfigurationError("non-finite JSON number: " + token)
        return value

    return json.loads(text, object_pairs_hook=unique_object,
                      parse_constant=reject_constant, parse_float=finite_float)


def validate(config):
    """Pure validation: no input, output, environment or scheduler access."""
    errors = []

    def section(value, name, keys):
        if type(value) is not dict:
            errors.append(name + " must be an object")
            value = {}
        for key in sorted(set(value) - set(keys)):
            errors.append(name + "." + key + ": unknown declaration")
        for key in keys:
            if key not in value:
                errors.append(name + "." + key + ": required declaration missing")
        return value

    def string(value, name):
        if type(value) is not str or not value.strip():
            errors.append(name + " must be a nonempty string")
            return ""
        return value

    def absolute(value, name):
        value = string(value, name)
        if (not value.startswith("/") or value.startswith("//")
                or any(part in (".", "..") for part in value.split("/"))
                or "//" in value or (value != "/" and value.endswith("/"))):
            errors.append(name + " must be an explicit normalized absolute POSIX path")
        if any(character in value for character in "*?[]"):
            errors.append(name + ": glob patterns are forbidden")
        return value

    def positive_integer(value, name):
        if type(value) is not int or value <= 0:
            errors.append(name + " must be a positive integer")

    def declarations(value, name):
        if type(value) is str:
            if "REQUIRES_OWNER_DECISION" in value:
                errors.append(name + ": REQUIRES_OWNER_DECISION is unresolved")
            if any(ord(c) < 32 or ord(c) == 127 for c in value):
                errors.append(name + ": control characters are forbidden")
            if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
                errors.append(name + ": unpaired Unicode surrogates are forbidden")
        elif type(value) is dict:
            for key, child in value.items():
                declarations(child, name + "." + key)
        elif type(value) is list:
            for index, child in enumerate(value):
                declarations(child, name + "[" + str(index) + "]")

    declarations(config, "config")
    root = section(config, "config", ("schema_version", "case", "historical",
                                       "runtime", "slurm", "audit", "approval",
                                       "diagnostics"))
    if type(root.get("schema_version")) is not int or root.get("schema_version") != 1:
        errors.append("schema_version must be integer 1")
    case = section(root.get("case"), "case", ("dataset_id", "field", "timestep",
                   "input_path", "input", "ordering_code", "ordering_label", "truncation"))
    for key in ("dataset_id", "field", "timestep"):
        string(case.get(key), "case." + key)
    input_path = absolute(case.get("input_path"), "case.input_path")
    # Do not infer field/dataset metadata; only verify the CLI timestep suffix.
    suffix = re.search(r"_([0-9]+)\.dat$", PurePosixPath(input_path).name)
    if suffix is None or suffix.group(1) != case.get("timestep"):
        errors.append("case.timestep must exactly match the _<digits>.dat input suffix")
    inp = section(case.get("input"), "case.input",
                  tuple(FIXED_INPUT) + ("crop_bounds", "physical_units"))
    for key, expected in FIXED_INPUT.items():
        actual = inp.get(key)
        matches = type(actual) is type(expected) and actual == expected
        if type(expected) is list:
            matches = matches and all(type(a) is type(b) for a, b in zip(actual, expected))
        if not matches:
            errors.append("case.input." + key + " must explicitly equal " + repr(expected))
    bounds = section(inp.get("crop_bounds"), "case.input.crop_bounds", ("x", "y", "z"))
    for axis, expected in (("x", [176, 688]), ("y", [248, 760]), ("z", [32, 544])):
        actual = bounds.get(axis)
        if (type(actual) is not list or actual != expected
                or any(type(v) is not int for v in actual)):
            errors.append("case.input.crop_bounds." + axis + " must equal " + repr(expected))
    string(inp.get("physical_units"), "case.input.physical_units")
    ordering = case.get("ordering_code")
    if type(ordering) is not int or ordering not in TAGS:
        errors.append("case.ordering_code must be exactly one integer in 1..4")
    elif case.get("ordering_label") != TAGS[ordering]:
        errors.append("case.ordering_label must match the literal historical tag " + TAGS[ordering])
    truncation = section(case.get("truncation"), "case.truncation", ("kind", "value"))
    if truncation.get("kind") != "max_bond_dimension":
        errors.append("Q04 unresolved: numeric cutoff configurations are unsupported in this "
                      "first interface; declare max_bond_dimension for chi=N. "
                      "Historical cutoff calculation remains unchanged.")
    cap = truncation.get("value")
    if type(cap) is not str or re.fullmatch(r"[+-]?[0-9]+", cap) is None:
        errors.append("case.truncation.value must be an integer token string for chi=N")
    else:
        try:
            int(cap)
        except ValueError:
            errors.append("case.truncation.value is not a representable planner integer token")
    historical = section(root.get("historical"), "historical", ("working_directory",))
    working = absolute(historical.get("working_directory"), "historical.working_directory")
    runtime = section(root.get("runtime"), "runtime", ("julia_executable", "omp_num_threads",
                                                        "julia_num_threads", "module_environment"))
    absolute(runtime.get("julia_executable"), "runtime.julia_executable")
    positive_integer(runtime.get("omp_num_threads"), "runtime.omp_num_threads")
    if runtime.get("julia_num_threads") != "unset":
        positive_integer(runtime.get("julia_num_threads"), "runtime.julia_num_threads")
    string(runtime.get("module_environment"), "runtime.module_environment")
    slurm_keys = ("cluster", "partition", "account", "qos", "reservation", "constraint",
                  "nodes", "ntasks", "cpus_per_task", "memory", "walltime", "job_name",
                  "stdout_path", "stderr_path")
    slurm = section(root.get("slurm"), "slurm", slurm_keys)
    for key in slurm_keys:
        if key in ("nodes", "ntasks", "cpus_per_task"):
            positive_integer(slurm.get(key), "slurm." + key)
        elif key in ("stdout_path", "stderr_path"):
            absolute(slurm.get(key), "slurm." + key)
        else:
            string(slurm.get(key), "slurm." + key)
    # Resource semantics/eligibility and task launch rules remain unverified.
    audit = section(root.get("audit"), "audit", ("directory",))
    audit_dir = absolute(audit.get("directory"), "audit.directory")
    approval = section(root.get("approval"), "approval", ("required_approver",))
    string(approval.get("required_approver"), "approval.required_approver")
    diagnostics = section(root.get("diagnostics"), "diagnostics", ("scope",))
    if diagnostics.get("scope") != "structural-and-run-only":
        errors.append("diagnostics.scope must be structural-and-run-only")
    if not errors:
        filename = ("truncated_" + PurePosixPath(input_path).name + "_512_"
                    + TAGS[ordering] + "_chi" + str(int(cap)) + ".mat")
        output_dir = PurePosixPath(working) / ("truncated_" + case["timestep"])
        output = output_dir / filename
        paths = [input_path, str(output), slurm["stdout_path"], slurm["stderr_path"]]
        if len(set(paths)) != len(paths):
            errors.append("input, output, stdout and stderr paths must be distinct")
        audit_path = PurePosixPath(audit_dir)
        if audit_path == output_dir or output_dir in audit_path.parents or audit_path in output_dir.parents:
            errors.append("audit.directory must be separate from the historical result directory")
        if audit_dir in paths:
            errors.append("audit.directory cannot be an input, output or log path")
        # Lexical checks only: never inspect the DNS path or existing results.
        files = [PurePosixPath(path) for path in paths]
        files.extend(audit_path / name for name in PROVENANCE_FILES.values())
        for index, first in enumerate(files):
            for second in files[index + 1:]:
                if first == second or first in second.parents or second in first.parents:
                    errors.append("declared file paths collide: " + str(first) + " and " + str(second))
        for directory in (PurePosixPath(working), output_dir, audit_path):
            for file_path in files:
                if file_path == directory or file_path in directory.parents:
                    errors.append("declared file path conflicts with a directory: " + str(file_path)
                                  + " and " + str(directory))
    if errors:
        raise ConfigurationError("\n".join(errors))
    return str(output)


def git_metadata():
    """Only read-only Git commands, never shell or workflow commands."""
    def git(*args):
        result = subprocess.run(["git", "--no-optional-locks", "-C", str(REPOSITORY), *args],
                                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                encoding="utf-8")
        return result.stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain")}


def build_report(config, raw_config):
    output = validate(config)
    case, runtime = config["case"], config["runtime"]
    argv = [runtime["julia_executable"], str(AUTHORITY), case["input_path"],
            "chi=" + case["truncation"]["value"], str(case["ordering_code"])]
    environment = {"OMP_NUM_THREADS": str(runtime["omp_num_threads"]),
                   "JULIA_NUM_THREADS": str(runtime["julia_num_threads"])}
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = lambda b: hashlib.sha256(b).hexdigest()
    audit = PurePosixPath(config["audit"]["directory"])
    report = {
        "state": "DRY RUN — NOT APPROVED — NOT SUBMITTED",
        "validation": "DECLARATIONS VALID; EXECUTION PREFLIGHT UNVERIFIED",
        "case": case,
        "working_directory": config["historical"]["working_directory"],
        "working_directory_note": "Required future process cwd; running the payload elsewhere changes its output path",
        "argv": argv,
        "historical_command": shlex.join(argv),
        "environment_declarations": environment,
        "module_environment_declaration": runtime["module_environment"],
        "environment_note": "unset requires absence at future launch; no environment is applied here",
        "historical_filename": PurePosixPath(output).name,
        "proposed_output_path": output,
        "proposed_provenance_paths": {key: str(audit / name)
                                      for key, name in PROVENANCE_FILES.items()},
        "requested_slurm_resources": config["slurm"],
        "diagnostics_scope": config["diagnostics"]["scope"],
        "required_approver": config["approval"]["required_approver"],
        "scope": {"cases": 1, "raw_values": 501645312, "historical_read_bytes": 2006581248,
                  "retained_values": 134217728, "retained_float32_bytes": 536870912,
                  "qubit_sites": 27, "calculations": 1, "reconstructions": 1, "MAT_writes": 1,
                  "runtime_estimate": "unknown", "peak_memory_estimate": "unknown"},
        "provenance": {
            "git": git_metadata(),
            "authority_path": str(AUTHORITY),
            "authority_sha256": digest(AUTHORITY.read_bytes()),
            "planner_sha256": digest(Path(__file__).read_bytes()),
            "original_config_sha256": digest(raw_config),
            "canonical_config_sha256": digest(canonical.encode("utf-8")),
            "canonical_config": canonical,
            "input_identity": "unverified; DNS input not opened, statted, resolved or hashed",
        },
        "unverified": [
            "Q01/Q11/Q17: input existence, readability, size, identity and symlink policy",
            "Q03: cap scientific suitability and target Julia Int representability",
            "Q07/Q08/Q09: runtime/packages, threads, resources and launch eligibility",
            "Q06/Q12/Q14: path permissions/collisions, revision approval and launch revalidation",
            "Q13/Q15: approval records and structural diagnostic collection",
        ],
        "limitations": [
            "Q04 unresolved: numeric cutoffs rejected temporarily; no cutoff-label table",
            "Literal historical ordering tags only; no scientific bit-label interpretation",
            "No files created, Julia executed, DNS read, Slurm queried/submitted, or metrics computed",
            "This proposal is not a complete CRC preflight or permission to run",
        ],
    }
    # Digest excludes itself; approval and future artifacts are not fabricated.
    report["proposal_sha256"] = digest(json.dumps(report, sort_keys=True,
                                      separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", help="one explicit JSON configuration")
    args = parser.parse_args(argv)
    try:
        raw = Path(args.configuration).read_bytes()
        config = parse_configuration(raw.decode("utf-8"))
        report = build_report(config, raw)
    except (ValueError, RecursionError, UnicodeError, OSError,
            subprocess.SubprocessError) as error:
        reason = str(error).encode("utf-8", errors="backslashreplace").decode("utf-8")
        print("DRY RUN — BLOCKED — NOT APPROVED — NOT SUBMITTED\n" + reason, file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
