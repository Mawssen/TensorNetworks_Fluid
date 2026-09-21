#!/bin/bash
#SBATCH --job-name=peps_inspect
#SBATCH --output=peps_inspect_%j.out
#SBATCH --error=peps_inspect_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --mem=4G
#SBATCH --time=0:05:00

# -----------------------------------------------------------------------------
# Diagnostic: inspect the PEPS stats pickle to confirm what got saved.
# Reads results/peps_0198/peps_stats_0198_wf_D=9_periodic_mixed_o4.pkl and
# prints ranges for the manifold subsample + joint-PDF grid stats.
#
# Submit:
#   sbatch submit_peps_inspect.sh
# View:
#   cat peps_inspect_<jobid>.out
# -----------------------------------------------------------------------------

source /ix/pgivi/moe32/envs/my_env/bin/activate
module load python/ondemand-jupyter-python3.11

cd "$SLURM_SUBMIT_DIR"

PKL_PATH="${PKL_PATH:-results/peps_0198/peps_stats_0198_wf_D=9_periodic_mixed_o4.pkl}"

echo "PKL_PATH = $PKL_PATH"
echo "which python: $(which python)"
echo ""

python << 'PYEOF'
import os
import pickle
import numpy as np

pkl = os.environ.get("PKL_PATH",
    "results/peps_0198/peps_stats_0198_wf_D=9_periodic_mixed_o4.pkl")
print(f"Reading pickle: {pkl}")

with open(pkl, "rb") as f:
    r = pickle.load(f)

print("Top-level keys:", sorted(r.keys()))
print()

# --- Manifold subsample -----------------------------------------------------
if "manifold_scatter" in r:
    print("=== manifold_scatter ===")
    m = r["manifold_scatter"]
    for k in ("Z", "Y_O2", "Y_OH", "T"):
        if k not in m:
            print(f"  {k}: (missing)")
            continue
        v = np.asarray(m[k])
        n = v.size
        frac_neg = (v < 0).sum() / n * 100
        frac_zero = (v == 0).sum() / n * 100
        frac_pos = (v > 0).sum() / n * 100
        print(f"  {k:5s}: min={v.min():+.4e}  max={v.max():+.4e}  mean={v.mean():+.4e}  N={n:,}")
        print(f"         frac<0: {frac_neg:5.2f}%  frac==0: {frac_zero:5.2f}%  frac>0: {frac_pos:5.2f}%")
    print()

# --- Joint PDF grid ---------------------------------------------------------
if "joint_pdf_Z_YCO2" in r:
    print("=== joint_pdf_Z_YCO2 ===")
    j = r["joint_pdf_Z_YCO2"]
    H = np.asarray(j["joint_pdf"])
    n = H.size
    finite = np.isfinite(H).sum()
    print(f"  joint_pdf: shape={H.shape}, dtype={H.dtype}")
    print(f"    finite fraction: {finite}/{n} = {finite/n*100:.1f}%")
    if finite > 0:
        Hf = H[np.isfinite(H)]
        print(f"    finite min={Hf.min():.3e}, max={Hf.max():.3e}, sum={Hf.sum():.3e}")
    print(f"  psi_Z    range: [{j['psi_Z'].min():.4f}, {j['psi_Z'].max():.4f}]  N={len(j['psi_Z'])}")
    print(f"  psi_YCO2 range: [{j['psi_YCO2'].min():.4e}, {j['psi_YCO2'].max():.4e}]  N={len(j['psi_YCO2'])}")
    print()

# --- Profiles (mean/RMS) ----------------------------------------------------
if "profiles" in r:
    print("=== profile ranges ===")
    for var in ("mixfrac", "T", "Y_CO", "Y_CO2", "Y_OH", "Y_O2", "chi"):
        if var not in r["profiles"]:
            continue
        p = r["profiles"][var]
        m = np.asarray(p["mean"])
        s = np.asarray(p["rms"])
        print(f"  {var:8s}  mean=[{m.min():+.3e}, {m.max():+.3e}]  "
              f"rms=[{s.min():+.3e}, {s.max():+.3e}]")
    print()

# --- Extinction summary -----------------------------------------------------
if "extinction" in r:
    print("=== extinction ===")
    for k, v in r["extinction"].items():
        print(f"  {k:>28s} = {v}")

print("\nDone.")
PYEOF
