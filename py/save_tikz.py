"""
save_tikz.py
------------
Thin wrapper around tikzplotlib for exporting matplotlib figures to .tex.

Usage:
    from save_tikz import save_tikz_if_available
    save_tikz_if_available(fig, "/path/to/output.tex")

If tikzplotlib is not installed, this becomes a no-op with a warning.
Install it once via:
    pip install tikzplotlib
"""

from __future__ import annotations

import os
from typing import Optional


_TIKZ_AVAILABLE: Optional[bool] = None
_TIKZ_MODULE = None


def _try_import():
    global _TIKZ_AVAILABLE, _TIKZ_MODULE
    if _TIKZ_AVAILABLE is not None:
        return

    # tikzplotlib 0.9.6 was last released in 2022 and calls several
    # matplotlib APIs that were renamed with underscore prefixes or
    # otherwise moved after matplotlib 3.8. Install shims for the ones
    # that come up in practice with our plots.
    #
    # Shim 1: common_texification -> _tex_escape (renamed in mpl 3.8)
    try:
        import matplotlib.backends.backend_pgf as _pgf
        if not hasattr(_pgf, "common_texification") and hasattr(_pgf, "_tex_escape"):
            _pgf.common_texification = _pgf._tex_escape
    except Exception:
        pass

    # Shim 2: Legend.legendHandles -> Legend.legend_handles (renamed in
    # mpl 3.9). tikzplotlib reads legend.legendHandles when serialising
    # legends; without this shim every plot with a legend fails export.
    try:
        from matplotlib.legend import Legend as _Legend
        if not hasattr(_Legend, "legendHandles"):
            def _legend_handles_getter(self):
                return self.legend_handles
            _Legend.legendHandles = property(_legend_handles_getter)
    except Exception:
        pass

    # Shim 3: Line2D._us_dashSeq -> Line2D._dash_pattern (or public dashes).
    # tikzplotlib reads the private _us_dashSeq attribute; matplotlib 3.10
    # removed it in favour of a public `_dash_pattern` tuple. Fall back to
    # the effective dash tuple from `get_linestyle()`.
    try:
        from matplotlib.lines import Line2D as _Line2D
        if not hasattr(_Line2D, "_us_dashSeq"):
            def _us_dashSeq_getter(self):
                # Prefer explicit dash pattern if available
                dp = getattr(self, "_dash_pattern", None)
                if dp is not None:
                    # _dash_pattern is a (offset, seq) tuple
                    try:
                        return dp[1]
                    except Exception:
                        return dp
                return None
            _Line2D._us_dashSeq = property(_us_dashSeq_getter)
        if not hasattr(_Line2D, "_us_dashOffset"):
            def _us_dashOffset_getter(self):
                dp = getattr(self, "_dash_pattern", None)
                if dp is not None:
                    try:
                        return dp[0]
                    except Exception:
                        return 0
                return 0
            _Line2D._us_dashOffset = property(_us_dashOffset_getter)
    except Exception:
        pass

    # Shim 4: np.float_ / np.int_ compat.
    # tikzplotlib 0.9.6 uses `np.float_` which was removed in NumPy 2.0.
    # Alias to np.float64. Same for np.int_ (removed in favour of np.int64).
    try:
        import numpy as _np
        if not hasattr(_np, "float_"):
            _np.float_ = _np.float64
        if not hasattr(_np, "int_"):
            _np.int_ = _np.int64
    except Exception:
        pass

    try:
        import tikzplotlib
        _TIKZ_MODULE = tikzplotlib
        _TIKZ_AVAILABLE = True
    except ImportError:
        _TIKZ_AVAILABLE = False
        print("[save_tikz] tikzplotlib not installed; skipping .tex output. "
              "Run `pip install tikzplotlib` in my_env to enable.")
    except Exception as e:
        _TIKZ_AVAILABLE = False
        print(f"[save_tikz] tikzplotlib import failed ({e.__class__.__name__}: "
              f"{e}); skipping .tex output.")


def save_tikz_if_available(fig, tex_path: str,
                            axis_width: str = r"\linewidth",
                            axis_height: str = r"0.7\linewidth") -> bool:
    """Save the matplotlib figure to a standalone tikz .tex file.

    Returns True if the file was written; False if tikzplotlib is missing
    or the save failed.
    """
    _try_import()
    if not _TIKZ_AVAILABLE:
        return False
    try:
        os.makedirs(os.path.dirname(tex_path) or ".", exist_ok=True)
        _TIKZ_MODULE.save(
            tex_path,
            figure=fig,
            axis_width=axis_width,
            axis_height=axis_height,
        )
        return True
    except Exception as e:
        print(f"[save_tikz] Failed to save {tex_path}: {e}")
        return False
