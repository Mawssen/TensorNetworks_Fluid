#!/bin/bash
#SBATCH --job-name=peps_orient
#SBATCH --output=peps_orient_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --mem=16G
#SBATCH --time=0:10:00

source /ix/pgivi/moe32/envs/my_env/bin/activate
module load python/ondemand-jupyter-python3.11

cd "$SLURM_SUBMIT_DIR"

python << 'PYEOF'
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dns_stats import GridConfig, load_field
from peps_io import load_peps_field

grid = GridConfig()

print("Loading DNS Z...")
dns_Z = load_field("/ix/pgivi/moe32/Aidyn_DNS/stats/jet_0198/jet_mixfrac_0198.dat",
                    grid).astype(np.float32)

print("Loading PEPS Z (with transpose(2,1,0) applied by peps_io)...")
peps_Z = load_peps_field(
    "/ix/pgivi/moe32/Aidyn_DNS/stats/truncated_0198/PEPS/mixfrac_wf_D=9_periodic.mat"
).astype(np.float32)

# Fix sign and rescale (2D-slice-level approximation is fine for the visual test)
Zf64 = peps_Z.astype(np.float64)
dnsf64 = dns_Z.astype(np.float64)
if np.sum(Zf64 * dnsf64) < 0:
    Zf64 = -Zf64
n_dns = np.sqrt(np.sum(dnsf64 ** 2))
n_peps = np.sqrt(np.sum(Zf64 ** 2))
if n_peps > 0:
    Zf64 = Zf64 * (n_dns / n_peps)
peps_Z = np.clip(Zf64, 0.0, 1.0).astype(np.float32)

mid = 256

dns_slab = dns_Z[:, :, mid]
peps_slab_raw = peps_Z[:, :, mid]

# 8 possible orientations of the PEPS slice
# 4 rotations x 2 (with or without flip)
variants = {}
variants["raw  (peps[:,:,mid])"]    = peps_slab_raw
variants["rot90(k=1)  CCW"]          = np.rot90(peps_slab_raw, k=1)
variants["rot90(k=2)  180"]          = np.rot90(peps_slab_raw, k=2)
variants["rot90(k=-1) CW"]            = np.rot90(peps_slab_raw, k=-1)
variants["transpose (.T)"]           = peps_slab_raw.T
variants["flipud (flip axis=0)"]     = np.flip(peps_slab_raw, axis=0)
variants["fliplr (flip axis=1)"]     = np.flip(peps_slab_raw, axis=1)
variants[".T then flipud"]           = np.flip(peps_slab_raw.T, axis=0)

# Draw grid: DNS + 8 PEPS variants, each rendered with the SAME imshow(arr.T, ...)
# call as plot_qc4pde uses. That way what you see IS what the paper contour panel
# would show if we applied that transformation.
n_variants = len(variants)
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
extent = [-3.55, 3.55, -3.55, 3.55]

# Top-left: DNS
axes[0, 0].imshow(dns_slab.T, origin='lower', cmap='inferno',
                    vmin=0, vmax=0.8, extent=extent, aspect='auto',
                    interpolation='nearest')
axes[0, 0].set_title("DNS (reference)", fontweight='bold')
axes[0, 0].set_xlabel("x/H"); axes[0, 0].set_ylabel("y/H")

# Remaining 8: PEPS variants
axes_flat = axes.ravel()
for i, (label, arr) in enumerate(variants.items(), start=1):
    ax = axes_flat[i]
    ax.imshow(arr.T, origin='lower', cmap='inferno',
               vmin=0, vmax=0.8, extent=extent, aspect='auto',
               interpolation='nearest')
    ax.set_title(label)
    ax.set_xlabel("x/H"); ax.set_ylabel("y/H")

plt.tight_layout()
outpath = "QC4PDE/peps_orient_check.png"
plt.savefig(outpath, dpi=120)
print(f"\nWrote {outpath}")
print("\nOpen this image and identify which panel matches DNS.")
PYEOF
