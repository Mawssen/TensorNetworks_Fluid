"""Lightweight routing tests: no FFTs, reductions, production I/O or jobs."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/python"))
from turbulence_tools_stats import api


class PreferredDefaultsTests(unittest.TestCase):
    def test_spectrum_uses_tested_float64_kernel(self):
        with patch.object(api, "calculate_spectrum_binned", return_value="new") as new, \
                patch.object(api, "_reference_module", side_effect=AssertionError("reference chosen")):
            self.assertEqual(api.calculate_spectrum("u", "v", "w"), "new")
            new.assert_called_once_with("u", "v", "w")

    def test_metrics_use_tested_chunked_kernel(self):
        with patch.object(api, "compare_fields_chunked", return_value="new") as new, \
                patch.object(api, "_reference_module", side_effect=AssertionError("reference chosen")):
            self.assertEqual(api.compare_fields("a", "b"), "new")
            new.assert_called_once_with("a", "b")

    def test_explicit_originals_remain_selectable(self):
        old = SimpleNamespace(calculate_spectrum=Mock(return_value="old"),
            calculate_fidelity=Mock(return_value=0.5), calculate_l2norm=Mock(return_value=0.25))
        with patch.object(api, "_reference_module", return_value=old), \
                patch.object(api, "calculate_spectrum_binned", side_effect=AssertionError("new chosen")), \
                patch.object(api, "compare_fields_chunked", side_effect=AssertionError("new chosen")):
            self.assertEqual(api.calculate_spectrum("u", "v", "w", implementation="reference"), "old")
            self.assertEqual(api.compare_fields("a", "b", implementation="reference"),
                             {"fidelity": 0.5, "relative_l2": 0.25})

    def test_original_derivative_stays_default(self):
        old = SimpleNamespace(spectral_derivative=Mock(return_value="old"))
        with patch.object(api, "_reference_module", return_value=old), \
                patch.object(api, "spectral_derivative_slabbed", return_value="slabbed") as slabbed:
            self.assertEqual(api.spectral_derivative("u", 1, 0.1), "old")
            old.spectral_derivative.assert_called_once_with("u", 1, 0.1, order=1)
            slabbed.assert_not_called()
            self.assertEqual(api.spectral_derivative("u", 1, 0.1, implementation="optimized"), "slabbed")

    def test_unknown_implementation_is_rejected(self):
        for call in (lambda: api.calculate_spectrum(None,None,None,implementation="float32"),
                     lambda: api.compare_fields(None,None,implementation="typo"),
                     lambda: api.spectral_derivative(None,0,1,implementation="typo")):
            with self.assertRaises(ValueError):
                call()

    def test_reference_loader_uses_preserved_sources(self):
        for name, function in (("calculate_spectrum.py", "calculate_spectrum"),
                               ("calculate_spectral_derivative.py", "spectral_derivative"),
                               ("calculate_u_comp_fidelity_l2norm.py", "calculate_fidelity")):
            self.assertTrue(callable(getattr(api._reference_module(name), function)))


if __name__ == "__main__":
    unittest.main()
