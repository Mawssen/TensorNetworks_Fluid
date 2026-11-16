"""Launch tests use a fake executable, never Julia or DNS data."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_mps_v1_dry_run import fixture, planner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mps_v1_run as runner


class ExecutionTests(unittest.TestCase):
    def setup_case(self, root, behavior="success"):
        config = fixture()
        inp = root / "example_0007.dat"
        inp.touch()  # Empty test file; no scientific input is simulated.
        config["case"]["input_path"] = str(inp)
        config["historical"]["working_directory"] = str(root)
        config["audit"]["directory"] = str(root / "audit")
        for name in ("stdout", "stderr"):
            config["slurm"][name + "_path"] = str(root / (name + ".log"))
        fake = root / "fake-julia"
        fake.write_text("#!" + sys.executable + "\n" +
            "import json, os, pathlib, sys\n"
            "print(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), "
            "'omp': os.getenv('OMP_NUM_THREADS'), 'threads': os.getenv('JULIA_NUM_THREADS')}))\n"
            "print('fake stderr', file=sys.stderr)\n" +
            ("sys.exit(7)\n" if behavior == "failure" else
             "pass\n" if behavior == "missing" else
             "p = pathlib.Path('truncated_0007/truncated_example_0007.dat_512_il_chi23.mat')\n"
             "p.parent.mkdir()\np.write_bytes(b'fake output')\n"))
        fake.chmod(0o700)
        config["runtime"]["julia_executable"] = str(fake)
        return config

    def invoke(self, config):
        with patch.dict(os.environ, SLURM_JOB_ID="test", JULIA_NUM_THREADS="99"), \
                patch.object(runner.socket, "gethostname", return_value="compute-test"):
            return runner.execute(config, json.dumps(config).encode())

    def test_actual_child_argv_cwd_environment_logs_and_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.setup_case(Path(temp))
            config["case"]["truncation"]["value"] = "+0023"
            status, result = self.invoke(config)
            self.assertEqual(status, 0)
            observed = json.loads(Path(config["slurm"]["stdout_path"]).read_text())
            self.assertEqual(observed["argv"], [str(planner.AUTHORITY),
                config["case"]["input_path"], "chi=+0023", "1"])
            self.assertEqual(observed["cwd"], temp)
            self.assertEqual(observed["omp"], "10")
            self.assertIsNone(observed["threads"])
            self.assertEqual(result["state"], "SUCCEEDED")
            self.assertEqual(result["output_bytes"], 11)
            self.assertIn("fake stderr", Path(config["slurm"]["stderr_path"]).read_text())
            self.assertTrue((Path(temp) / "audit/run.json").is_file())

    def test_child_failure_and_missing_output(self):
        for behavior, expected in (("failure", 7), ("missing", 1)):
            with self.subTest(behavior=behavior), tempfile.TemporaryDirectory() as temp:
                status, result = self.invoke(self.setup_case(Path(temp), behavior))
                self.assertEqual(status, expected)
                self.assertEqual(result["state"], "FAILED")

    def test_explicit_threads(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.setup_case(Path(temp))
            config["runtime"]["julia_num_threads"] = 2
            self.invoke(config)
            self.assertEqual(json.loads(Path(config["slurm"]["stdout_path"]).read_text())["threads"], "2")

    def test_existing_output_logs_and_audit_refused(self):
        for collision in ("output", "stdout", "audit"):
            with self.subTest(collision=collision), tempfile.TemporaryDirectory() as temp:
                config = self.setup_case(Path(temp))
                if collision == "output":
                    path = Path(planner.validate(config))
                    path.parent.mkdir()
                    path.write_text("preserve")
                elif collision == "stdout":
                    path = Path(config["slurm"]["stdout_path"])
                    path.write_text("preserve")
                else:
                    path = Path(config["audit"]["directory"])
                    path.mkdir()
                with self.assertRaises((ValueError, OSError)):
                    self.invoke(config)

    def test_login_node_blocked_even_with_job_id(self):
        config = fixture()
        with patch.dict(os.environ, SLURM_JOB_ID="test"), \
                patch.object(runner.socket, "gethostname", return_value="login1.crc.pitt.edu"), \
                patch.object(runner.subprocess, "run", wraps=runner.subprocess.run):
            with self.assertRaisesRegex(ValueError, "compute allocation"):
                runner.execute(config, json.dumps(config).encode())

    def test_launch_oserror_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.setup_case(Path(temp))
            with patch.object(runner, "build_report", return_value=planner.build_report(config, json.dumps(config).encode())), \
                    patch.object(runner.subprocess, "run", side_effect=OSError("launch failed")):
                status, result = self.invoke(config)
            self.assertEqual(status, 1)
            self.assertIn("launch failed", result["error"])


if __name__ == "__main__":
    unittest.main()
