"""Synthetic spectrum checks; FFTs run only within a compute allocation."""

import importlib.util
from pathlib import Path
import os
import socket
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/python"))
from turbulence_tools_stats import calculate_spectrum_binned
from turbulence_tools_stats.spectrum import _shell_geometry, _shell_indices

SPEC = importlib.util.spec_from_file_location(
    "historical_spectrum", ROOT / "legacy/mps_py_codes/calculate_spectrum.py")
historical = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(historical)

COMPUTE = bool(os.environ.get("SLURM_JOB_ID")) and not socket.gethostname().startswith("login")


class SpectrumDeclarationTests(unittest.TestCase):
    def test_invalid_declarations_do_not_start_fft(self):
        u = np.zeros((2, 2, 2), dtype=np.float64)
        examples = [((u.astype(np.float32), u, u), {}, TypeError),
                    ((u[:, :, 0], u, u), {}, ValueError),
                    ((np.zeros((2, 2, 3)), u, u), {}, ValueError),
                    ((np.zeros((1, 1, 1)), u, u), {}, ValueError),
                    ((np.zeros((3, 3, 3)), u, u), {}, ValueError),
                    ((u, u, u), {"slab_size": 0}, ValueError),
                    ((u, u, u), {"slab_size": 1.5}, TypeError),
                    ((u, u, u), {"slab_size": True}, TypeError)]
        with patch.object(np.fft, "fftn", side_effect=AssertionError("FFT started")):
            for args, kwargs, error in examples:
                with self.subTest(kwargs=kwargs, error=error), self.assertRaises(error):
                    calculate_spectrum_binned(*args, **kwargs)


@unittest.skipUnless(COMPUTE, "numerical checks require a compute allocation")
class SpectrumNumericalTests(unittest.TestCase):
    def test_shell_membership_and_counts_match_actual_reference(self):
        digitize = np.digitize
        for dim in (2, 7, 8, 16):
            captured = []

            def capture(*args, **kwargs):
                result = digitize(*args, **kwargs)
                captured.append(result.copy())
                return result

            u = np.zeros((dim, dim, dim), dtype=np.float64)
            with patch.object(np, "digitize", side_effect=capture):
                k_reference, _ = historical.calculate_spectrum(u, u, u)
            self.assertEqual(len(captured), 1)
            rx, k, bins, dx = _shell_geometry(dim)
            np.testing.assert_array_equal(k, k_reference)
            for slab in (1, 3, dim + 1):
                chunks = [_shell_indices(rx, bins, dx, start, min(start + slab, dim))
                          for start in range(0, dim, slab)]
                actual = np.concatenate(chunks, axis=0)
                np.testing.assert_array_equal(actual, captured[0])
                count = np.bincount(actual.ravel(), minlength=len(k) + 2)[1:-1]
                expected = [np.count_nonzero(captured[0] == i + 1) for i in range(len(k))]
                np.testing.assert_array_equal(count, expected)

    def test_constants_and_impulses_match_exactly(self):
        dim = 8
        fields = [np.zeros((dim, dim, dim), dtype=np.float64) for _ in range(3)]
        for constant in (0.0, 1.0, 3.0):
            u = np.full((dim, dim, dim), constant, dtype=np.float64)
            k, result = calculate_spectrum_binned(u, u, u, slab_size=3)
            expected_k, expected = historical.calculate_spectrum(u, u, u)
            np.testing.assert_array_equal(k, expected_k)
            np.testing.assert_array_equal(result, expected)
        for component in range(3):
            fields[component][0, 0, 0] = float(component + 1)
        before = [field.copy() for field in fields]
        k, result = calculate_spectrum_binned(*fields, slab_size=3)
        expected_k, expected = historical.calculate_spectrum(*fields)
        np.testing.assert_array_equal(k, expected_k)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.dtype(np.float64))
        for field, original in zip(fields, before):
            np.testing.assert_array_equal(field, original)

    def test_noncontiguous_fields_and_slabs(self):
        fields = [np.zeros((8, 8, 8), dtype=np.float64) for _ in range(3)]
        fields[0][0, 0, 0] = 1.0
        fields[1][4, 0, 0] = 2.0
        fields = [field.transpose(2, 0, 1) for field in fields]
        expected_k, expected = historical.calculate_spectrum(*fields)
        for slab in (1, 3, 8, 17):
            k, result = calculate_spectrum_binned(*fields, slab_size=slab)
            np.testing.assert_array_equal(k, expected_k)
            np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
