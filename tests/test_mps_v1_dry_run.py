"""Text-only tests: invented metadata, no DNS files or scientific fixtures."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("mps_v1_dry_run", ROOT / "scripts/mps_v1_dry_run.py")
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


def fixture():
    """Synthetic declaration values only; never a production case selection."""
    config = json.loads((ROOT / "docs/mps_v1_dry_run_template.json").read_text())
    config["case"].update(dataset_id="synthetic-metadata", field="example",
                          timestep="0007", input_path="/unverified/example_0007.dat",
                          ordering_code=1, ordering_label="il")
    config["case"]["input"]["physical_units"] = "synthetic-unit"
    config["case"]["truncation"]["value"] = "23"
    config["historical"]["working_directory"] = "/unverified/run"
    config["runtime"].update(julia_executable="/unverified/bin/julia", omp_num_threads=10,
                             julia_num_threads="unset", module_environment="unset")
    config["slurm"].update(cluster="example", partition="example", account="unset", qos="unset",
                           reservation="unset", constraint="unset", nodes=1, ntasks=1,
                           cpus_per_task=10, memory="owner-declared-limit", walltime="owner-declared-time",
                           job_name="example", stdout_path="/unverified/log/stdout",
                           stderr_path="/unverified/log/stderr")
    config["audit"]["directory"] = "/unverified/audit"
    config["approval"]["required_approver"] = "synthetic-owner"
    return config


def change(config, path, value):
    obj = config
    parts = path.split(".")
    for key in parts[:-1]:
        obj = obj[key]
    obj[parts[-1]] = value


class DeclarationTests(unittest.TestCase):
    def test_valid_declarations_require_no_input_or_output_access(self):
        with patch.object(Path, "open", side_effect=AssertionError("filesystem access")), \
                patch.object(Path, "stat", side_effect=AssertionError("stat access")), \
                patch.object(planner.subprocess, "run", side_effect=AssertionError("process access")):
            output = planner.validate(fixture_without_file_access())
        self.assertEqual(output, "/unverified/run/truncated_0007/truncated_example_0007.dat_512_il_chi23.mat")

    def test_missing_every_required_key(self):
        config = fixture()

        def visit(obj, prefix=""):
            for key, value in obj.items():
                path = prefix + key
                if type(value) is dict:
                    yield from visit(value, path + ".")
                yield path

        for path in visit(config):
            with self.subTest(path=path):
                candidate = copy.deepcopy(config)
                obj = candidate
                parts = path.split(".")
                for key in parts[:-1]:
                    obj = obj[key]
                del obj[parts[-1]]
                with self.assertRaises(planner.ConfigurationError):
                    planner.validate(candidate)

    def test_invalid_contract_and_selector_values(self):
        examples = {
            "schema_version": [True, "1", 2],
            "case.dataset_id": ["", " ", ["a", "b"]],
            "case.field": [None, ["u", "v"]],
            "case.timestep": [7, "7", "0008"],
            "case.input_path": ["relative_0007.dat", "/a/../example_0007.dat", "/a/file.bin",
                                "/a/example*_0007.dat"],
            "case.ordering_code": [True, "1", 0, 5, [1, 2]],
            "case.ordering_label": ["seq", "interleaved"],
            "case.input.domain": ["other"],
            "case.input.scalar_type": ["Float64", "float32"],
            "case.input.endianness": ["little-endian"],
            "case.input.full_grid_shape": [[576, 1008, 864], [864.0, 1008, 576]],
            "case.input.axis_order": [["z", "y", "x"]],
            "case.input.fastest_axis": ["z"],
            "case.input.crop_kind": ["corner"],
            "case.input.crop_shape": [[256, 256, 256]],
            "case.input.crop_indexing": ["one-based-inclusive"],
            "case.input.crop_bounds.x": [[177, 688], [176.0, 688]],
            "case.input.crop_bounds.y": [[0, 512]],
            "case.input.crop_bounds.z": [[32, 543]],
            "case.input.physical_units": ["", " ", 1],
            "case.truncation.value": [23, 2.3, "chi=23", "NaN", "1e-4", ["23", "24"]],
            "historical.working_directory": ["relative"],
            "runtime.julia_executable": ["julia"],
            "runtime.omp_num_threads": [True, 0],
            "runtime.julia_num_threads": [True, "10"],
            "slurm.nodes": [0, True],
            "audit.directory": ["relative", "/unverified/run/truncated_0007/audit"],
            "diagnostics.scope": ["fidelity"],
        }
        for path, values in examples.items():
            for value in values:
                with self.subTest(path=path, value=value):
                    candidate = fixture()
                    change(candidate, path, value)
                    with self.assertRaises(planner.ConfigurationError):
                        planner.validate(candidate)

    def test_cutoffs_rejected_with_q04_without_formatting(self):
        for kind in ("cutoff", "4.1e-4", 0.001):
            candidate = fixture()
            candidate["case"]["truncation"] = {"kind": kind, "value": "0.001"}
            with self.assertRaisesRegex(planner.ConfigurationError, "Q04 unresolved"):
                planner.validate(candidate)

    def test_placeholder_unknown_keys_controls_and_collisions(self):
        for path, value in (("case.field", "REQUIRES_OWNER_DECISION"),
                            ("runtime.module_environment", "module load x\nrun"),
                            ("slurm.stderr_path", "/unverified/log/stdout")):
            candidate = fixture()
            change(candidate, path, value)
            with self.assertRaises(planner.ConfigurationError):
                planner.validate(candidate)
        for section, key in (("slurm", "array"), ("case", "normalization"),
                             ("historical", "julia_script")):
            candidate = fixture()
            candidate[section][key] = "unsupported"
            with self.assertRaisesRegex(planner.ConfigurationError, "unknown declaration"):
                planner.validate(candidate)
        with self.assertRaises(planner.ConfigurationError):
            planner.validate(json.loads((ROOT / "docs/mps_v1_dry_run_template.json").read_text()))

    def test_duplicate_keys_invalid_json_and_nonfinite_values(self):
        for text in ('{"case": {}, "case": {}}', '{"x": NaN}', '{"x": Infinity}',
                     '{"x": 1e999}', '{"x": -1e999}'):
            with self.assertRaises(planner.ConfigurationError):
                planner.parse_configuration(text)
        with self.assertRaises(json.JSONDecodeError):
            planner.parse_configuration('{"case":')

    def test_invalid_root_and_section_objects(self):
        for candidate in (None, [], "config", 1):
            with self.subTest(candidate=candidate), self.assertRaises(planner.ConfigurationError):
                planner.validate(candidate)
        for name in ("case", "runtime", "slurm", "audit", "approval", "diagnostics"):
            candidate = fixture()
            candidate[name] = []
            with self.subTest(name=name), self.assertRaises(planner.ConfigurationError):
                planner.validate(candidate)

    def test_all_supported_tags_and_integer_label_canonicalization(self):
        for code, tag in planner.TAGS.items():
            candidate = fixture()
            candidate["case"].update(ordering_code=code, ordering_label=tag)
            candidate["case"]["truncation"]["value"] = "+0023"
            self.assertTrue(planner.validate(candidate).endswith("_" + tag + "_chi23.mat"))

    def test_chi_domain_remains_unverified_not_silently_selected(self):
        for token, label in (("0", "0"), ("-0023", "-23"), ("+0023", "23"),
                             (str(2 ** 100), str(2 ** 100))):
            candidate = fixture()
            candidate["case"]["truncation"]["value"] = token
            self.assertTrue(planner.validate(candidate).endswith("_chi" + label + ".mat"))

    def test_provenance_and_directory_collisions(self):
        for path, value in (("slurm.stdout_path", "/unverified/audit/plan.json"),
                            ("case.input_path", "/unverified/audit/plan.json/example_0007.dat"),
                            ("slurm.stderr_path", "/unverified/log/stdout/child"),
                            ("slurm.stdout_path", "/unverified/run"),
                            ("slurm.stdout_path", "/unverified/audit"),
                            ("slurm.stderr_path", "/unverified/run/truncated_0007")):
            candidate = fixture()
            change(candidate, path, value)
            with self.subTest(path=path, value=value), \
                    self.assertRaisesRegex(planner.ConfigurationError, "collid|conflict|cannot"):
                planner.validate(candidate)

    def test_invalid_unicode_declaration_fails_clearly(self):
        candidate = fixture()
        candidate["case"]["input"]["physical_units"] = "unit\ud800"
        with self.assertRaisesRegex(planner.ConfigurationError, "Unicode surrogates"):
            planner.validate(candidate)


# Construct before the safety mocks; this reads only the JSON template.
PURE_FIXTURE = fixture()


def fixture_without_file_access():
    return copy.deepcopy(PURE_FIXTURE)


class ReportTests(unittest.TestCase):
    def test_report_quotes_command_and_performs_only_approved_source_reads(self):
        candidate = fixture()
        candidate["case"]["input_path"] = "/unverified/a '$(echo no)'_0007.dat"
        allowed = {str(planner.AUTHORITY), str(Path(planner.__file__))}
        original_open = io.open

        def guarded_open(path, *args, **kwargs):
            self.assertIn(str(path), allowed)
            return original_open(path, *args, **kwargs)

        with patch.object(planner, "git_metadata", return_value={"commit": "test", "status": ""}), \
                patch("io.open", side_effect=guarded_open), \
                patch.object(planner.subprocess, "run", side_effect=AssertionError("payload execution")):
            raw = json.dumps(candidate).encode()
            report = planner.build_report(candidate, raw)
        self.assertEqual(shlex.split(report["historical_command"]), report["argv"])
        self.assertEqual(report["argv"][1], str(planner.AUTHORITY))
        self.assertEqual(report["argv"], [candidate["runtime"]["julia_executable"],
                         str(planner.AUTHORITY), candidate["case"]["input_path"], "chi=23", "1"])
        self.assertIn("unverified", report["provenance"]["input_identity"])
        self.assertEqual(len(report["proposal_sha256"]), 64)
        self.assertEqual(report["proposed_provenance_paths"]["plan"], "/unverified/audit/plan.json")

    def test_main_success_and_blocked_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "case.json"
            path.write_text(json.dumps(fixture()))
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(planner, "git_metadata", return_value={"commit": "test", "status": ""}), \
                    contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(planner.main([str(path)]), 0)
            self.assertIn("DECLARATIONS VALID", json.loads(stdout.getvalue())["validation"])
            candidate = fixture()
            candidate["case"]["truncation"]["kind"] = "cutoff"
            path.write_text(json.dumps(candidate))
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(planner.main([str(path)]), 2)
            self.assertIn("Q04 unresolved", stderr.getvalue())

    def test_git_metadata_only_uses_read_only_git(self):
        with patch.object(planner.subprocess, "run") as run:
            run.return_value.stdout = "fixture"
            planner.git_metadata()
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertEqual(call.args[0][:2], ["git", "--no-optional-locks"])
            self.assertNotIn("shell", call.kwargs)

    def test_working_directory_is_declared_not_inherited_or_applied(self):
        candidate = fixture()
        candidate["historical"]["working_directory"] = "/other owner's/run"
        candidate["case"]["truncation"]["value"] = "+0023"
        with patch.object(planner, "git_metadata", return_value={"commit": "test", "status": ""}), \
                patch.object(Path, "cwd", side_effect=AssertionError("cwd inference")), \
                patch.object(planner.subprocess, "run", side_effect=AssertionError("execution")):
            report = planner.build_report(candidate, json.dumps(candidate).encode())
        self.assertEqual(report["working_directory"], "/other owner's/run")
        self.assertEqual(report["proposed_output_path"],
                         "/other owner's/run/truncated_0007/truncated_example_0007.dat_512_il_chi23.mat")
        self.assertEqual(report["argv"][-2:], ["chi=+0023", "1"])
        self.assertIn("Q03", " ".join(report["unverified"]))

    def test_crop_bounds_match_historical_centering_formula(self):
        inp = fixture()["case"]["input"]
        for axis, size in zip(inp["axis_order"], inp["full_grid_shape"]):
            # Julia's one-based start is div(size,2)-div(512,2)+1.
            start = size // 2 - 512 // 2
            self.assertEqual(inp["crop_bounds"][axis], [start, start + 512])

    def test_main_rejects_malformed_text_without_traceback_or_provenance(self):
        candidate = fixture()
        candidate["case"]["field"] = "\ud800"
        texts = [json.dumps(candidate), "[" * 2000 + "]" * 2000,
                 '{"x": 1e999}', '{"case":']
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            for text in texts:
                path.write_text(text)
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.object(planner, "git_metadata", side_effect=AssertionError("provenance")), \
                        contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    self.assertEqual(planner.main([str(path)]), 2)
                self.assertEqual(stdout.getvalue(), "")
                self.assertIn("DRY RUN — BLOCKED", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_main_only_reads_configuration_and_approved_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "case.json"
            path.write_text(json.dumps(fixture()))
            allowed = {str(path), str(planner.AUTHORITY), str(Path(planner.__file__))}
            original_open = io.open

            def guarded_open(open_path, *args, **kwargs):
                self.assertIn(str(open_path), allowed)
                self.assertEqual(args[0], "rb")
                return original_open(open_path, *args, **kwargs)

            with patch("io.open", side_effect=guarded_open), \
                    patch.object(Path, "stat", side_effect=AssertionError("filesystem preflight")), \
                    patch.object(planner, "git_metadata", return_value={"commit": "test", "status": ""}), \
                    patch.object(planner.subprocess, "run", side_effect=AssertionError("execution")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(planner.main([str(path)]), 0)

    def test_success_with_real_local_git_provenance(self):
        candidate = fixture()
        report = planner.build_report(candidate, json.dumps(candidate).encode())
        self.assertRegex(report["provenance"]["git"]["commit"], r"^[0-9a-f]{40}$")
        self.assertRegex(report["provenance"]["authority_sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
