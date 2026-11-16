# PeleLM-FDF DNS statistics pipeline

Reproduces the DNS statistics from Aitzhan et al., *Combustion Theory and
Modelling* (2022) for a single snapshot of the CO/H2 temporally evolving
jet flame, using a 512^3 center cube extracted from the 864x1008x576 DNS.

## Files

| File                   | Purpose                                                |
|------------------------|--------------------------------------------------------|
| `dns_stats.py`         | Core module: field loader + one function per statistic |
| `run_dns_stats.py`     | Driver: loads fields, computes every stat, saves `.pkl` + `.npz` |
| `plot_dns_stats.py`    | Reads saved results, produces figures                  |
| `submit_dns_stats.sh`  | SLURM submission script                                |
| `smoke_test.py`        | End-to-end sanity check on tiny synthetic fields       |

## Running on the cluster

Edit the paths at the top of `submit_dns_stats.sh`:

```bash
DATA_DIR=/ix/pgivi/moe32/DNS       # folder with jet_<var>_0228.dat
TIMESTEP=0228
OUT_DIR=./results/dns_0228
```

Submit:

```bash
sbatch submit_dns_stats.sh
```

Expected runtime: tens of minutes at most (all stats are one or two
passes over 512^3 float32 arrays). Memory: each field is 512 MB; with 14
fields loaded the peak is ~7 GB, well under the 64 GB requested.

## Statistics computed

Every DNS curve from the paper's figures is reproduced:

| Figure | Statistic                                                  |
|--------|------------------------------------------------------------|
| 6      | Mean + RMS profiles of Z, T, Y_CO, Y_CO2 vs y/H            |
| 8      | Mixture-fraction thickness delta_Z/(2H) (single value)     |
| 9      | Scalar dissipation rate profile chi(y/H)                   |
| 10     | Conditional mean temperature E(T | Z = psi_Z)              |
| 11     | Volume-averaged extinction marker + T on stoich. surface   |
| 12     | Marginal PDF of Z near centerline                          |
| 13     | Joint PDF of (Z, Y_CO2) near centerline                    |
| 14     | Compositional manifold scatter (Z, Y_O2, Y_OH) colored by T|

Figure 3 (temperature snapshot) and Figure 7 (instantaneous slices) are
not reproduced here -- they are visualizations of the raw field, easy to
generate directly with matplotlib if wanted.

Figures 4, 5 (Eulerian-vs-Lagrangian consistency) do not apply to this
DNS-only analysis.

Figures 8 and 11 are temporal plots in the paper; from a single snapshot
we contribute a single point each.

## Extending to MPS (next step)

The `dns_stats.py` module is agnostic about the field source. When the
MPS-truncated `.dat` files are ready:

1. Create a sibling `mps_stats.py` (or add a `source` parameter) that
   points `load_all_dns_fields` at the MPS directory with the correct
   naming convention.
2. Run `run_dns_stats.py` a second time on the MPS data -> second `.pkl`.
3. Write `plot_comparison.py` that loads both pickles and overlays the
   curves (DNS as solid, MPS as dashed/symbols), plus error metrics
   (profile L2 norm, PDF KL divergence, etc.).

Because each statistic function returns numpy arrays, a direct
difference (`stat_DNS - stat_MPS`) gives the error profile, and
`|diff|_2 / |stat_DNS|_2` gives a dimensionless fidelity metric matching
the style of the existing `calculate_fidelity` / `calculate_l2norm`
functions in the user's workflow.
