"""
aggregate_scatter_metrics.py
----------------------------
Walk a scatter-plot directory, gather per-task metrics JSONs, and produce
a per-field summary as both Excel (.xlsx) and CSV.

Excel layout (one workbook per field):
  Sheet 'RMS'        - 4 sp rows x 4 cutoff cols, color-scaled
  Sheet 'rel_L2'     - 4 sp rows x 4 cutoff cols, color-scaled
  Sheet 'corr'       - 4 sp rows x 4 cutoff cols, color-scaled (reverse)
  Sheet 'long'       - flat table of every (sp, cf) record
  Sheet 'meta'       - what modes / fd_orders / periodic settings appeared

CSV is the same long-format table -- handy for piping into pandas later.

Usage
-----
python aggregate_scatter_metrics.py <scatter_dir>

Example
-------
python aggregate_scatter_metrics.py results/plots_field_0198/scatter
"""

from __future__ import annotations

import os
import sys
import json
import glob
from typing import Dict, List

import pandas as pd

# Order of axes in the table
_SP_ORDER = [1, 2, 3, 4]
_CF_ORDER = ["1e-2", "1e-3", "1e-4", "1e-5"]
_ORD_NAME = {1: "il", 2: "seq", 3: "comb1", 4: "combn"}


def _load_records(scatter_dir: str) -> Dict[str, List[dict]]:
    """Group all .json metric files by field."""
    by_field: Dict[str, List[dict]] = {}
    for jp in sorted(glob.glob(os.path.join(scatter_dir, "*.json"))):
        try:
            with open(jp) as f:
                rec = json.load(f)
        except Exception as e:
            print(f"[skip] {jp}: {e}")
            continue
        if "field" not in rec:
            continue
        by_field.setdefault(rec["field"], []).append(rec)
    return by_field


def _grid_df(records: List[dict], metric: str) -> pd.DataFrame:
    """Build a (sp x cf) DataFrame for a given metric."""
    val: Dict[tuple, float] = {(r["sp"], r["cf_label"]): r.get(metric)
                               for r in records}
    rows = []
    for sp in _SP_ORDER:
        row = {"sp": f"sp{sp} ({_ORD_NAME[sp]})"}
        for cf in _CF_ORDER:
            row[cf] = val.get((sp, cf))
        rows.append(row)
    df = pd.DataFrame(rows).set_index("sp")
    return df


def _flat_df(records: List[dict]) -> pd.DataFrame:
    """Flat table: one row per record, sorted by (sp, cf)."""
    cols = ["field", "sp", "ordering", "cf_label", "mode", "fd_order",
            "periodic_xz_cube", "rms", "rel_l2", "corr",
            "n_total", "n_sampled", "png"]
    rows = []
    for r in records:
        rows.append({c: r.get(c) for c in cols})
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df = df.sort_values(["sp", "cf_label"]).reset_index(drop=True)
    return df


def _meta_df(records: List[dict]) -> pd.DataFrame:
    """Metadata summary -- what configurations were found."""
    modes = sorted({r.get("mode") for r in records if r.get("mode")})
    fd_orders = sorted({r.get("fd_order") for r in records
                        if r.get("fd_order") is not None})
    periodics = sorted({bool(r.get("periodic_xz_cube"))
                        for r in records if "periodic_xz_cube" in r})
    return pd.DataFrame([
        {"key": "field",         "value": records[0].get("field", "")},
        {"key": "n_cases",       "value": len(records)},
        {"key": "modes",         "value": ", ".join(map(str, modes))},
        {"key": "fd_orders",     "value": ", ".join(map(str, fd_orders))
                                            if fd_orders else "(N/A)"},
        {"key": "periodic_xz",   "value": ", ".join(map(str, periodics))},
    ])


def _write_field_summary(field: str, records: List[dict], out_dir: str):
    xlsx_path = os.path.join(out_dir, f"scatter_summary_{field}.xlsx")
    csv_path = os.path.join(out_dir, f"scatter_summary_{field}.csv")

    rms_df = _grid_df(records, "rms")
    l2_df = _grid_df(records, "rel_l2")
    corr_df = _grid_df(records, "corr")
    long_df = _flat_df(records)
    meta_df = _meta_df(records)

    # CSV: just the long table, easy for scripting
    long_df.to_csv(csv_path, index=False, float_format="%.6e")
    print(f"[write] {csv_path}")

    # Excel layout:
    #   Sheets RMS / rel_L2 / corr  -- plain grids, no color
    #   Sheet 'compare_by_cutoff'   -- single combined view, ranked across
    #                                  the 4 sp values within each cutoff
    #                                  for each metric (per-column ranking)
    #   Sheets long, meta
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for name, df in [("RMS", rms_df), ("rel_L2", l2_df), ("corr", corr_df)]:
            df.to_excel(writer, sheet_name=name)
        long_df.to_excel(writer, sheet_name="long", index=False)
        meta_df.to_excel(writer, sheet_name="meta", index=False)

        # Format the plain grid sheets (number format, column width) but
        # apply NO conditional color formatting.
        try:
            wb = writer.book
            for sheet_name in ("RMS", "rel_L2", "corr"):
                ws = wb[sheet_name]
                fmt = "0.000E+00" if sheet_name != "corr" else "0.00000"
                for row in ws.iter_rows(min_row=2, max_row=5,
                                        min_col=2, max_col=5):
                    for cell in row:
                        cell.number_format = fmt
                for col_letter in ["A", "B", "C", "D", "E"]:
                    ws.column_dimensions[col_letter].width = 14
        except Exception as e:
            print(f"[warn] could not format Excel grid sheets: {e}")

        # Build the 'compare_by_cutoff' sheet: stacked metric blocks, with
        # color applied PER COLUMN (cutoff), so within each cutoff column
        # the 4 sp variants are ranked best -> worst.
        try:
            _write_compare_sheet(writer, rms_df, l2_df, corr_df)
        except Exception as e:
            print(f"[warn] could not build compare_by_cutoff sheet: {e}")

    print(f"[write] {xlsx_path}")


def _write_compare_sheet(writer, rms_df, l2_df, corr_df):
    """Build 'compare_by_cutoff' sheet.

    Layout:
        Header row 1: blank, then 4 cutoff labels
        Header row 2 ('Metric: RMS'):
        Row per sp variant -> values
        Blank
        Header row ('Metric: rel_L2'):
        Row per sp variant -> values
        Blank
        Header row ('Metric: corr'):
        Row per sp variant -> values

    Each metric block has color applied PER COLUMN, with green = best
    and red = worst within that column. RMS and rel_L2 use lower-is-better;
    corr uses higher-is-better.
    """
    from openpyxl.formatting.rule import ColorScaleRule

    sheet_name = "compare_by_cutoff"
    # Write skeleton via pandas: simplest is a single long DataFrame with
    # a synthetic index, write it, then apply per-column color rules to
    # the 3 metric blocks separately.
    blocks = [
        ("RMS",    rms_df,  False, "0.000E+00"),
        ("rel_L2", l2_df,   False, "0.000E+00"),
        ("corr",   corr_df, True,  "0.00000"),
    ]
    rows = []
    rows.append({"label": "", **{c: c for c in rms_df.columns}})  # not used
    # Build via writer using openpyxl directly so we control row positions
    wb = writer.book
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(sheet_name)

    # Column widths
    ws.column_dimensions["A"].width = 22
    for letter in ["B", "C", "D", "E"]:
        ws.column_dimensions[letter].width = 14

    # Header: cutoffs across columns B..E
    ws.cell(row=1, column=1, value="").font = None
    for j, cf in enumerate(rms_df.columns, start=2):
        ws.cell(row=1, column=j, value=cf)

    cur_row = 2
    block_ranges = []   # to apply color rules at the end
    for metric_name, df, higher_is_better, num_fmt in blocks:
        # Section header
        ws.cell(row=cur_row, column=1, value=f"Metric: {metric_name}")
        cur_row += 1
        block_first = cur_row
        for sp_label, vals in df.iterrows():
            ws.cell(row=cur_row, column=1, value=sp_label)
            for j, cf in enumerate(df.columns, start=2):
                v = vals[cf]
                if v is None or (isinstance(v, float) and v != v):  # NaN
                    ws.cell(row=cur_row, column=j, value=None)
                else:
                    cell = ws.cell(row=cur_row, column=j, value=float(v))
                    cell.number_format = num_fmt
            cur_row += 1
        block_last = cur_row - 1
        block_ranges.append((metric_name, block_first, block_last,
                             higher_is_better))
        cur_row += 1   # blank row between blocks

    # Apply color rules PER COLUMN within each metric block.
    # ColorScaleRule applied to a single-column range gives a per-column
    # ranking automatically.
    for metric_name, r_start, r_end, higher_is_better in block_ranges:
        if r_end < r_start:
            continue
        if higher_is_better:
            # higher = green
            rule = ColorScaleRule(
                start_type="min", start_color="F8696B",
                mid_type="percentile", mid_value=50, mid_color="FFEB84",
                end_type="max", end_color="63BE7B",
            )
        else:
            # lower = green
            rule = ColorScaleRule(
                start_type="min", start_color="63BE7B",
                mid_type="percentile", mid_value=50, mid_color="FFEB84",
                end_type="max", end_color="F8696B",
            )
        # One rule per column, range B<r_start>:B<r_end> etc.
        for col_letter in ["B", "C", "D", "E"]:
            rng = f"{col_letter}{r_start}:{col_letter}{r_end}"
            ws.conditional_formatting.add(rng, rule)


def main():
    if len(sys.argv) != 2:
        print("Usage: python aggregate_scatter_metrics.py <scatter_dir>")
        sys.exit(1)
    scatter_dir = sys.argv[1]
    if not os.path.isdir(scatter_dir):
        print(f"Not a directory: {scatter_dir}")
        sys.exit(1)

    by_field = _load_records(scatter_dir)
    if not by_field:
        print(f"No metric JSONs found in {scatter_dir}")
        sys.exit(1)

    print(f"Found metrics for {len(by_field)} fields: "
          f"{sorted(by_field.keys())}")
    for field, records in sorted(by_field.items()):
        _write_field_summary(field, records, scatter_dir)


if __name__ == "__main__":
    main()
