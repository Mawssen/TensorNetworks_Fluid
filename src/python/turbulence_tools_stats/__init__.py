"""Preferred statistics API plus explicit alternatives; no Julia/Python bridge."""

from .spectrum import calculate_spectrum_binned
from .reductions import spectral_derivative_slabbed, compare_fields_chunked
from .api import calculate_spectrum, compare_fields, spectral_derivative

__all__ = ["calculate_spectrum", "compare_fields", "spectral_derivative",
           "calculate_spectrum_binned", "spectral_derivative_slabbed", "compare_fields_chunked"]
