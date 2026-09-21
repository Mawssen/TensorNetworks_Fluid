"""
run_peps_stats.py
-----------------
Compute the same stats pipeline for PEPS-truncated fields that
run_mps_stats.py computes for MPS. Produces a pickle that plugs into
plot_dns_mps_comparison.py as another "MPS-like" source (label 'peps').

Usage
-----
    python run_peps_stats.py <peps_data_dir> <timestep> <out_dir> \
        [--tag wf_D=9_periodic] \
        [--z-chi-mode {mixed,recon}] \
        [--fd-order {2,4,6,8}] \
        [--periodic-xz {true,false}]

Notes
-----
    - PEPS files store the field under key 'Z' (not 'u' as MPS).
    - No stored chi in PEPS output; chi is always reconstructed from
      Z and alpha via compute_chi.
    - z_chi_mode 'stored' is not available (nothing stored). Default is
      'mixed', matching MPS default when comparing to LES.
"""

from __future__ import annotations

import os
import sys
import time
import pickle
from typing import Optional

import numpy as np

from dns_stats import GridConfig, compute_all_stats, Z_ST_DEFAULT
from derived_fields import compute_chi, compute_Z_bilger, StreamConfig
from peps_io import load_all_peps_fields, peps_case_label


def _parse(args, flag, default=None, cast=str):
    if flag in args:
        i = args.index(flag)
        v = args[i + 1]
        args.pop(i + 1); args.pop(i)
        return cast(v) if cast is not str else v
    return default


def main():
    args = sys.argv[1:]
    tag = _parse(args, "--tag", default="wf_D=9_periodic")
    z_chi_mode = _parse(args, "--z-chi-mode", default="mixed")
    fd_order = int(_parse(args, "--fd-order", default="4"))
    periodic_str = _parse(args, "--periodic-xz", default="false")
    # Optional: directory of DNS binary fields. If given, each PEPS field is
    # auto-rescaled by matching its L2 norm to the DNS L2 norm (undoes
    # PEPS's unit-norm state normalization). Skip via --no-rescale.
    dns_data_dir = _parse(args, "--dns-data-dir",
                          default="/ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198")
    no_rescale = "--no-rescale" in args
    if no_rescale:
        args.remove("--no-rescale")
    periodic_xz_cube = periodic_str.lower() in ("true", "1", "yes", "on")

    if len(args) != 3:
        print("Usage: python run_peps_stats.py <peps_data_dir> <timestep> <out_dir> "
              "[--tag TAG] [--z-chi-mode {mixed,recon}] "
              "[--fd-order {2,4,6,8}] [--periodic-xz {true,false}]")
        sys.exit(1)
    if z_chi_mode not in ("mixed", "recon"):
        print(f"Invalid --z-chi-mode {z_chi_mode}; use 'mixed' or 'recon'.")
        sys.exit(1)
    if fd_order not in (2, 4, 6, 8):
        print(f"Invalid --fd-order {fd_order}")
        sys.exit(1)

    data_dir, timestep, out_dir = args
    os.makedirs(out_dir, exist_ok=True)

    grid = GridConfig()
    label = peps_case_label(tag=tag)

    t0 = time.time()
    print(f"[peps] data_dir={data_dir}  timestep={timestep}  tag={tag}",
          flush=True)
    print(f"[peps] z_chi_mode={z_chi_mode}  fd_order={fd_order}  "
          f"periodic_xz_cube={periodic_xz_cube}", flush=True)

    fields = load_all_peps_fields(data_dir, tag=tag)
    print(f"[peps] loaded {len(fields)} fields in {time.time()-t0:.1f}s. "
          f"Keys: {sorted(fields.keys())}", flush=True)
    for k, v in fields.items():
        print(f"    {k:>10s}: shape={v.shape}, "
              f"min={float(v.min()):+.3e}, max={float(v.max()):+.3e}",
              flush=True)

    # ----- Auto-rescale by DNS L2 norm ---------------------------------
    # PEPS commonly stores each field as a unit-norm state (||psi||_2 = 1).
    # Undo by scaling each PEPS field so ||PEPS||_2 == ||DNS||_2 for the
    # same variable. Fields whose norm already matches DNS (e.g. alpha)
    # are left untouched -- the trigger is a ratio > 2x or < 0.5x.
    if not no_rescale:
        from dns_stats import load_field
        # DNS variable-name mapping (PEPS key -> DNS filename var token).
        _DNS_VAR = {"mixfrac": "mixfrac", "T": "T", "alpha": "alpha",
                    "Y_CO": "Y_CO", "Y_CO2": "Y_CO2", "Y_OH": "Y_OH",
                    "Y_O2": "Y_O2", "Y_H2": "Y_H2", "Y_H2O": "Y_H2O"}
        print(f"[peps] auto-rescaling fields to match DNS L2 norms and sign "
              f"(dns_data_dir={dns_data_dir})...", flush=True)
        for peps_key, dns_var in _DNS_VAR.items():
            if peps_key not in fields:
                continue
            dns_path = os.path.join(dns_data_dir, f"jet_{dns_var}_{timestep}.dat")
            if not os.path.exists(dns_path):
                print(f"    {peps_key:>10s}: DNS file missing at {dns_path}; "
                      f"skipping rescale", flush=True)
                continue
            t_r = time.time()
            dns_arr = load_field(dns_path, grid).astype(np.float32)
            peps_arr = fields[peps_key]

            # Sign fix: PEPS extracts each field from a tensor network up to
            # an arbitrary global sign. Compare the sign of <peps, dns> (dot
            # product) to decide whether to flip. This is more robust than
            # comparing min/max because it uses the entire field.
            dot = float(np.sum(peps_arr.astype(np.float64)
                                * dns_arr.astype(np.float64)))
            sign = -1.0 if dot < 0 else 1.0
            if sign < 0:
                peps_arr = peps_arr * sign
                fields[peps_key] = peps_arr

            norm_dns = float(np.sqrt(np.sum(dns_arr.astype(np.float64) ** 2)))
            del dns_arr
            norm_peps = float(np.sqrt(np.sum(peps_arr.astype(np.float64) ** 2)))
            if norm_peps <= 0.0:
                print(f"    {peps_key:>10s}: PEPS norm is 0; skipping",
                      flush=True)
                continue
            ratio = norm_dns / norm_peps
            sign_note = "  [SIGN FLIPPED]" if sign < 0 else ""
            if 0.5 < ratio < 2.0 and sign > 0:
                print(f"    {peps_key:>10s}: norm ratio DNS/PEPS = {ratio:.3f}"
                      f"{sign_note}  (already scale-matched, no rescale)",
                      flush=True)
                continue
            fields[peps_key] = (peps_arr * ratio).astype(np.float32)
            new_min = float(fields[peps_key].min())
            new_max = float(fields[peps_key].max())
            print(f"    {peps_key:>10s}: ratio={ratio:.3e}{sign_note}  "
                  f"new range=[{new_min:+.3e}, {new_max:+.3e}]  "
                  f"({time.time()-t_r:.1f}s)", flush=True)

    # Clip PEPS fields to physical ranges. Truncation at D=9 introduces
    # small negative values for near-zero species (Y_CO2, Y_OH, Y_O2) and
    # for the low-Z tail of mixture fraction. Left alone, these break
    # histograms and joint PDFs whose fixed axis ranges start at 0.
    # T is clipped from below at a small positive value to keep chi and
    # log-scaled plots happy; mixfrac and species are clipped to [0, ~ub]
    # matching their DNS-side conventional range.
    _CLIP_RANGES = {
        "mixfrac": (0.0, 1.0),
        "Y_CO":    (0.0, 1.0),
        "Y_CO2":   (0.0, 1.0),
        "Y_OH":    (0.0, 1.0),
        "Y_O2":    (0.0, 1.0),
        "Y_H":     (0.0, 1.0),
        "Y_H2":    (0.0, 1.0),
        "Y_H2O":   (0.0, 1.0),
        "T":       (200.0, 3000.0),
        "alpha":   (0.0, 1e-2),
    }
    print(f"[peps] Clipping fields to physical ranges (avoids NaN in "
          f"joint PDFs / negative species)...", flush=True)
    for key, (lo, hi) in _CLIP_RANGES.items():
        if key not in fields:
            continue
        arr = fields[key]
        n_below = int((arr < lo).sum())
        n_above = int((arr > hi).sum())
        if n_below == 0 and n_above == 0:
            continue
        total = arr.size
        frac_lo = n_below / total * 100
        frac_hi = n_above / total * 100
        fields[key] = np.clip(arr, lo, hi).astype(np.float32)
        print(f"    {key:>10s}: clipped {n_below:,} cells below "
              f"({frac_lo:.3f}%) and {n_above:,} above ({frac_hi:.3f}%) "
              f"to [{lo}, {hi}]", flush=True)

    # PEPS files don't include rho; needed for compute_chi API. It gets
    # cancelled algebraically anyway (compute_chi divides by rho when
    # alpha is thermal diffusivity, so rho drops out). Use ones as a
    # neutral placeholder.
    if "rho" not in fields:
        fields["rho"] = np.ones_like(fields["mixfrac"], dtype=np.float32)

    # Reconstruct Z via Bilger from species if requested.
    if z_chi_mode == "recon":
        needed = ("Y_CO", "Y_CO2", "Y_H2", "Y_H2O", "Y_O2", "Y_OH")
        if all(k in fields for k in needed):
            print("[peps] reconstructing Z (Bilger) from PEPS species...",
                  flush=True)
            Z_rec, info = compute_Z_bilger(fields, streams=StreamConfig())
            fields["mixfrac"] = Z_rec
            print(f"       sum_Y mean={info['sumY_mean']:.4f}, "
                  f"clip frac={info['Z_clip_fraction']*100:.3f}%",
                  flush=True)
        else:
            missing = [k for k in needed if k not in fields]
            print(f"[peps] WARN: --z-chi-mode recon needs species; missing "
                  f"{missing}. Falling back to loaded mixfrac.", flush=True)

    # chi is always reconstructed from Z + alpha for PEPS.
    if "alpha" not in fields:
        print("[peps] ERROR: alpha field required to compute chi and not "
              "present. Add alpha_wf_...mat to the PEPS directory.",
              flush=True)
        sys.exit(1)
    print(f"[peps] computing chi = 2*alpha*|grad Z|^2 (FD order={fd_order}"
          f", periodic_xz={periodic_xz_cube})...", flush=True)
    chi_rec = compute_chi(
        fields["mixfrac"], fields["alpha"], fields["rho"],
        grid.dx_m, grid.dy_m, grid.dz_m,
        alpha_is="thermal_diffusivity", order=fd_order,
        periodic_xz=periodic_xz_cube,
    )
    fields["chi"] = chi_rec
    print(f"       chi: peak={float(chi_rec.max()):.3e}, "
          f"mean={float(chi_rec.mean()):.3e}", flush=True)

    t1 = time.time()
    print("[peps] computing stats...", flush=True)
    results = compute_all_stats(fields, grid, z_st=Z_ST_DEFAULT)
    print(f"[peps] stats done in {time.time()-t1:.1f}s", flush=True)

    ext = results["extinction"]
    print(f"\n=== [peps {label} mode={z_chi_mode}] Summary ===")
    print(f"delta_Z / (2H)        = {results['delta_Z_over_2H']:.4f}")
    print(f"volume-averaged M_ext = {ext['volume_averaged_marker']:.3e}")
    print(f"cond. P(extinction)   = {ext['conditional_extinction_prob']:.3e}")
    print(f"T on stoich surface   = {ext['T_on_stoich_surface']:.2f} K")
    print(f"n (near-stoich cells) = {ext['n_near_st_cells']:,} / "
          f"{ext['n_total_cells']:,}")

    # Metadata for the plotter to identify the PEPS source.
    results["peps_tag"] = tag
    results["peps_label"] = label
    results["z_chi_mode"] = z_chi_mode
    results["fd_order"] = fd_order
    results["periodic_xz_cube"] = periodic_xz_cube

    stem = f"peps_stats_{timestep}_{tag}_{z_chi_mode}_o{fd_order}"
    if periodic_xz_cube:
        stem += "_p"
    pkl_path = os.path.join(out_dir, stem + ".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[peps] save {pkl_path}")

    print(f"\nTotal wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
