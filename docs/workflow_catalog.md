# Repository workflow catalog for CRC standardization

Catalog date: 2026-09-21. This is a static, read-only inventory of the repository at commit `ee7ec85`. No historical program, notebook, Slurm file, dataset, result, environment, or job was executed or changed while preparing it.

## Scope, interpretation, and status labels

A **workflow** here is a distinct user-invokable computation, validation, diagnostic, or reporting path. Helper modules are listed with their caller rather than treated as independent workflows. Closely related plotting commands are grouped when they consume the same products and answer the same question; every executable entry point is nevertheless named below.

Status means:

- **Known working:** retained repository evidence records a completed run, not merely a runnable-looking script.
- **Structurally complete; execution unverified:** entry point, inputs, computation, and outputs are present, but this repository retains no run result or test log proving the path.
- **Incomplete in this checkout:** a required local file, portable path, or compatible handoff is missing.
- **Experimental:** one-off study, diagnostic, notebook, or hard-coded research case rather than a stable interface.
- **Uncertain:** intent or compatibility cannot be established without choosing among conflicting historical conventions.

“First clean workflow” is an assessment, not a selection of an authoritative scientific convention. **Yes** means the path is a good engineering target using its already-evidenced behavior. **Conditional** means it becomes a good target only after the listed scientific/compatibility decision. **No** means another workflow should establish shared conventions first.

## Repository-wide findings that affect every candidate

- There is no Python lock/requirements file and no Julia `Project.toml`/`Manifest.toml`. The modern Python wrappers repeatedly activate `/ix/pgivi/moe32/envs/my_env`, load `python/ondemand-jupyter-python3.11`, and sometimes `texlive/2021`; representative evidence is [`sh/submit_dns_stats.sh:19`](../sh/submit_dns_stats.sh#L19). The code additionally requires some combination of NumPy, SciPy, h5py, pandas, openpyxl, matplotlib, yt, tikzplotlib, cmasher, numba, and sympy. Exact versions are not evidenced.
- Many wrappers invoke `python script.py` or `julia script.jl` by basename without changing into this repository's `py/` or `Julia/` directory. They appear copied from external working directories such as `/ix/pgivi/moe32/Aidyn_DNS/stats`; for example, [`sh/submit_mps_stats.sh:89`](../sh/submit_mps_stats.sh#L89) calls a basename although the checked-in script lives under `py/`. Thus “wrapper present” does not mean “submit from repository root.”
- The modern jet-flame path consistently documents raw fields as big-endian `float32`, full shape `(nz,ny,nx)=(576,1008,864)`, returned as `(x,y,z)`, with a centered `512^3` crop ([`py/dns_stats.py:11`](../py/dns_stats.py#L11), [`py/dns_stats.py:119`](../py/dns_stats.py#L119)). This is repository evidence, not an endorsement of that convention for a future standard.
- MPS reconstruction files are MATLAB v7.3/HDF5 with datasets `u` and `chi_crit`; the current consumer expects names like `truncated_jet_<var>_<ts>.dat_512_<ordering>_cf<cutoff>.mat` or `_chi<N>.mat` ([`py/mps_io.py:1`](../py/mps_io.py#L1), [`py/mps_io.py:77`](../py/mps_io.py#L77)). PEPS files use per-field dataset names and `<var>_wf_D=9_periodic.mat` ([`py/peps_io.py:1`](../py/peps_io.py#L1)).
- The repository contains conflicting historical MPS implementations and conventions. The provisional-reference documents deliberately leave tensor bit significance, normalization, truncation/fidelity meaning, environment, first dataset, regression outputs, and tolerances unresolved ([`docs/mps_v1_decisions.md:13`](mps_v1_decisions.md#L13)). No entry below silently resolves them.
- The only retained batch-run ledger is for the CRC LES box-filter interface: `LES_interface/crc/les_history.json` contains 29 `done` records, with inputs, filter factors, timings, physical checks, outputs, and job IDs. Other “working” language in comments/docstrings is treated as a claim, not run proof.

## A. MPS and tensor-network workflows

### A1. Provisional jet-field MPS compression (`0mps_calc`)

- **Purpose:** Decode a full jet-flame `.dat` field, center-crop `512^3`, tensorize it under a selected split/order, build and truncate an MPS by cutoff or maximum bond dimension, reconstruct the field, and save link dimensions.
- **Entry scripts and local imports:** numerical entry [`Julia/0mps_calc.jl:806`](../Julia/0mps_calc.jl#L806); CRC array wrapper [`Julia/00mps_calc.slurm:1`](../Julia/00mps_calc.slurm#L1). There are no local Julia imports. Read-only support utilities are [`scripts/capture_mps_reference_environment.sh:1`](../scripts/capture_mps_reference_environment.sh#L1) and [`scripts/inspect_mps_reference_contract.jl:1`](../scripts/inspect_mps_reference_contract.jl#L1).
- **Language/environment:** Julia; `ITensors`, `ITensorMPS`, `LinearAlgebra`, `MAT`, `HDF5`, `Random`, `JSON`, `Printf`, `MKL`; BLAS threads set to 10 ([`Julia/0mps_calc.jl:1`](../Julia/0mps_calc.jl#L1)). Wrapper points to Julia `1.12.6` and exports `OMP_NUM_THREADS=10` ([`Julia/00mps_calc.slurm:18`](../Julia/00mps_calc.slurm#L18)). Package versions/project are absent.
- **Inputs:** positional `<input_path> <cutoff|chi=N> <split>`; basename must end `_<digits>.dat`. Active loader hard-codes `864x1008x576`, big-endian `Float32`, centered `512^3` ([`Julia/0mps_calc.jl:806`](../Julia/0mps_calc.jl#L806), [`Julia/0mps_calc.jl:825`](../Julia/0mps_calc.jl#L825)). Wrapper uses `./jet_<ts>/jet_<field>_<ts>.dat` for 14 timestep-0198 fields ([`Julia/00mps_calc.slurm:21`](../Julia/00mps_calc.slurm#L21), [`Julia/00mps_calc.slurm:42`](../Julia/00mps_calc.slurm#L42)).
- **Outputs/metrics:** `truncated_<ts>/truncated_<input-basename>_512_<tag>_<cf|chi>.mat`, datasets `u` and `chi_crit`; stdout timing. No fidelity or error metric is computed ([`Julia/0mps_calc.jl:416`](../Julia/0mps_calc.jl#L416), [`Julia/0mps_calc.jl:852`](../Julia/0mps_calc.jl#L852)).
- **Slurm:** 1 node/task, 20 CPUs, SMP partition, 64 GiB, 2 h, array `0-27%10` ([`Julia/00mps_calc.slurm:2`](../Julia/00mps_calc.slurm#L2)). The comment says four mixfrac tasks while active arrays define 28 multi-field cases; neither is selected as future intent.
- **Status:** **Structurally complete; execution unverified; scientifically unresolved.** The active call path is documented as a provisional reference, not validated as authoritative ([`docs/mps_v1_reference_contract.md:3`](mps_v1_reference_contract.md#L3)). Output directory layout also does not create the `sp<N>` subdirectories expected by newer stats wrappers.
- **First clean workflow:** **Conditional, high value but high risk.** It is central to later MPS work, but cannot be the first numerical rebuild until ordering/bit meaning, crop/layout, compression, environment, outputs, and tolerance are approved.

### A2. Key-value MPS reconstruction for JHS or jet velocity

- **Purpose:** Reconstruct one `.mat` or jet `.dat` velocity/scalar cube with a fixed `chi`, cutoff, and split; this is the source path used by several hand-written and generated component jobs.
- **Entry scripts and local imports:** [`Julia/mps_calc.jl:250`](../Julia/mps_calc.jl#L250), with no local imports. Wrappers/generators: [`Slurms/calc.slurm:1`](../Slurms/calc.slurm#L1), [`sh/submit_mps_chi9.sh:1`](../sh/submit_mps_chi9.sh#L1), [`sh/job_mps27_U.sh:1`](../sh/job_mps27_U.sh#L1), [`sh/job_mps27_V.sh:1`](../sh/job_mps27_V.sh#L1), [`sh/job_mps27_W.sh:1`](../sh/job_mps27_W.sh#L1), [`sh/job_mps29_u.sh:1`](../sh/job_mps29_u.sh#L1), [`sh/job_mps29_v.sh:1`](../sh/job_mps29_v.sh#L1), [`sh/job_mps29_w.sh:1`](../sh/job_mps29_w.sh#L1), [`sh/submit_all.sh:1`](../sh/submit_all.sh#L1), [`sh/submit_all2.sh:1`](../sh/submit_all2.sh#L1), [`py/gen_mps_chi_jobs.py:1`](../py/gen_mps_chi_jobs.py#L1), and older duplicate [`py/gen_mps_chi9_jobs.py:1`](../py/gen_mps_chi9_jobs.py#L1).
- **Language/environment:** Julia with `ITensors`, `ITensorMPS`, `LinearAlgebra`, `MAT`, `HDF5`, `Plots`, `MKL` ([`Julia/mps_calc.jl:1`](../Julia/mps_calc.jl#L1)). Generated jobs and hand wrappers use `$HOME/.juliaup/bin/julia`, a private per-job depot, and comments identify Julia 1.12; `Slurms/calc.slurm` instead sources `.bashrc`. Generator itself is standard-library Python.
- **Inputs:** key-value arguments `chi=`, `cutoff=`, `split=`, `file=`, `dir=`, `ext=`, optional `filepath=`. `.mat` takes the first dataset; `.dat` uses the same evidenced jet full-grid decode and centered crop ([`Julia/mps_calc.jl:94`](../Julia/mps_calc.jl#L94), [`Julia/mps_calc.jl:269`](../Julia/mps_calc.jl#L269)). JHS patterns are `{U,V,W}_t1.mat`; jet patterns are `jet_{u,v,w}_0198.dat`.
- **Outputs/metrics:** `truncated_<file>_1024_<il|sp>_chi<N>.mat`, datasets `u` and `chi_crit`, plus load statistics and timing ([`Julia/mps_calc.jl:299`](../Julia/mps_calc.jl#L299), [`Julia/mps_calc.jl:304`](../Julia/mps_calc.jl#L304)). Despite a jet `.dat` becoming `512^3`, the filename still says `1024`; this does not match `mps_io`'s current jet filename convention.
- **Slurm:** generic `calc.slurm`: 20 CPUs, 256 GiB, 23 h; component jobs: 4-8 CPUs, 100-200 GiB, 3 h; `submit_mps_chi9.sh` runs all three components serially and continues after failures.
- **Status:** **Structurally complete but compatibility-uncertain.** Multiple wrappers point to external copies of `mps_calc.jl`, and naming/handoff differs from the modern stats loader. There is no retained result ledger.
- **First clean workflow:** **Conditional.** Useful after A1 conventions are resolved; not suitable as a separate authority.

### A3. MPS cutoff-to-compression-ratio sweep

- **Purpose:** Build interleaved MPS representations over a cutoff list without reconstruction, record link dimensions and memory compression ratio, bracket a target CR, and optionally distribute one cutoff per CRC job.
- **Entry scripts and local imports:** [`Julia/mps_cutoff_sweep.jl:78`](../Julia/mps_cutoff_sweep.jl#L78); job generator [`py/gen_cutoff_jobs.py:1`](../py/gen_cutoff_jobs.py#L1); post-hoc collector [`py/measure_cr_sweep.py:1`](../py/measure_cr_sweep.py#L1). No local imports.
- **Language/environment:** Julia with `ITensors`, `ITensorMPS`, `MAT`, `MKL`, 20 BLAS threads; generator is standard-library Python; collector needs h5py, NumPy, and optional openpyxl.
- **Inputs:** JHS-style `<dir>/<file>.mat`, first dataset, power-of-two cubic field; CLI `file=`, `dir=`, `target=`, `out=`, comma-separated `cutoffs=` ([`Julia/mps_cutoff_sweep.jl:94`](../Julia/mps_cutoff_sweep.jl#L94)). Collector scans `truncated_jet_mixfrac_*.mat` and reads `chi_crit` ([`py/measure_cr_sweep.py:95`](../py/measure_cr_sweep.py#L95)).
- **Outputs/metrics:** text table of cutoff, `CR_memory`, max link dimension, and bond count; generated `.slurm` files plus `submit_all.sh`/collector; post-hoc CSV/XLSX summary. The CR formula is explicitly encoded at [`Julia/mps_cutoff_sweep.jl:55`](../Julia/mps_cutoff_sweep.jl#L55) and [`py/measure_cr_sweep.py:32`](../py/measure_cr_sweep.py#L32).
- **Slurm:** generated defaults are 20 CPUs, 180 GiB, 1 h, SMP/short, one cutoff per job ([`py/gen_cutoff_jobs.py:106`](../py/gen_cutoff_jobs.py#L106)). No checked-in static wrapper or retained run log.
- **Status:** **Structurally complete; execution unverified; convention-conditional.** It hard-codes split 1 and one CR definition.
- **First clean workflow:** **Conditional/strong for a low-output MPS pilot.** The post-hoc collector is low cost; the MPS-building sweep is costly. Standardization still requires approval of ordering and CR definition.

### A4. MPS truncation quality aggregation

- **Purpose:** Scan existing jet-field reconstruction MAT files and compare each against DNS, reporting maximum `chi_crit`, memory CR, cosine fidelity, and relative L2 in an Excel workbook.
- **Entry scripts and local imports:** [`py/analyze_mps_truncations.py:167`](../py/analyze_mps_truncations.py#L167); wrapper [`sh/submit_mps_analysis.sh:1`](../sh/submit_mps_analysis.sh#L1). No local imports.
- **Language/environment:** Python/NumPy/h5py/pandas and an Excel writer in `my_env`; Python 3.11 module and optional texlive are loaded.
- **Inputs:** `truncated_jet_<var>_<ts>.dat_512_{il|seq|comb1|combn}_cf*.mat` plus matching `jet_<var>_<ts>.dat` ([`py/analyze_mps_truncations.py:46`](../py/analyze_mps_truncations.py#L46)). The executable loader is centered in all axes ([`py/analyze_mps_truncations.py:80`](../py/analyze_mps_truncations.py#L80)); the earlier docstring's “sx=0” statement conflicts with that code.
- **Outputs/metrics:** one XLSX sheet per cutoff plus `all`; CR, fidelity, relative L2, max bond dimension ([`py/analyze_mps_truncations.py:4`](../py/analyze_mps_truncations.py#L4), [`py/analyze_mps_truncations.py:235`](../py/analyze_mps_truncations.py#L235)). Regex excludes `_chi<N>` cases.
- **Slurm:** 4 CPUs, 64 GiB, 2 h, SMP ([`sh/submit_mps_analysis.sh:2`](../sh/submit_mps_analysis.sh#L2)).
- **Status:** **Structurally complete; execution unverified; internally documented crop conflict.**
- **First clean workflow:** **Conditional.** Attractive read-mostly post-processing, but fidelity, CR, crop, and accepted filename families must first be agreed.

### A5. PEPS-to-MPS ordering/infidelity experiment

- **Purpose:** For a serialized small tensor (`c1_Re=20000_Nx=12_Ny=12.bin`), derive MPS bond caps from PEPS bond dimensions, sweep four orderings and seven caps, reconstruct, and measure overlap infidelity.
- **Entry scripts and local imports:** active block in [`Julia/mps_calc2.jl:713`](../Julia/mps_calc2.jl#L713). No local imports.
- **Language/environment:** Julia with `ITensors`, `ITensorMPS`, `MAT`, `HDF5`, `Random`, `JSON`, `Printf`, `Serialization`; version/project unrecorded.
- **Inputs:** Julia `deserialize` file in the working directory; hard-coded filename and `chi_peps=[8,16,24,32,40,48,64]` ([`Julia/mps_calc2.jl:720`](../Julia/mps_calc2.jl#L720)).
- **Outputs/metrics:** one reconstructed MAT (`u`, `chi_crit`) per ordering/cap and `infidelity_<file>_mps.mat` ([`Julia/mps_calc2.jl:724`](../Julia/mps_calc2.jl#L724)).
- **Slurm:** none evidenced.
- **Status:** **Experimental; execution unverified.** Hard-coded input, no CLI, and ordering/fidelity conventions are scientific decisions.
- **First clean workflow:** **No.** Preserve as an experiment until a user-selected reference case and conventions exist.

### A6. MPS mutual-information map

- **Purpose:** Build an untruncated MPS for JHS `U_t1`, reconstruct it, calculate pairwise site mutual information, label sites by ordering, and render a heatmap.
- **Entry scripts and local imports:** active block [`Julia/mps_calc3.jl:799`](../Julia/mps_calc3.jl#L799). No local imports.
- **Language/environment:** Julia packages from A5 plus `Plots` and `MKL`; BLAS threads 10 ([`Julia/mps_calc3.jl:1`](../Julia/mps_calc3.jl#L1)).
- **Inputs:** hard-coded `../../../../../Forced_Isotropic_1024Cubed_mat_in_one/U_t1.mat`, first dataset, `cf=0`, `sp=1` ([`Julia/mps_calc3.jl:802`](../Julia/mps_calc3.jl#L802)).
- **Outputs/metrics:** `MI_U_sp1_interleaved_cf_0.0.mat` and PNG; elapsed mutual-information time.
- **Slurm:** none evidenced.
- **Status:** **Experimental and computationally hazardous.** Pairwise reduced-density calculations on a 1024³ state are not evidenced as completed.
- **First clean workflow:** **No.** It depends on unresolved tensor-site semantics and has high cost.

### A7. Full bond singular-value extraction for JHS velocity

- **Purpose:** Tensorize each `1024^3` JHS velocity component using split 1 and save the normalized singular-value vector at every MPS bond.
- **Entry scripts and local imports:** [`Julia/1sv_U.jl:453`](../Julia/1sv_U.jl#L453), [`Julia/2sv_V.jl:453`](../Julia/2sv_V.jl#L453), [`Julia/3sv_W.jl:453`](../Julia/3sv_W.jl#L453); wrappers [`Slurms/11sv_U.slurm:1`](../Slurms/11sv_U.slurm#L1), `22sv_V.slurm`, `33sv_W.slurm`. No local imports.
- **Language/environment:** Julia `1.10.4`; ITensors, LinearAlgebra, MAT, HDF5, Plots, JSON, Printf, MKL; BLAS/OMP 20.
- **Inputs:** relative `../../../../Forced_Isotropic_1024Cubed_mat_in_one/{U,V,W}_t1.mat`, first dataset.
- **Outputs/metrics:** `full_singular_values_<component>_t1.mat`, key `sv`; printed singular-value lengths and timing ([`Julia/1sv_U.jl:453`](../Julia/1sv_U.jl#L453)).
- **Slurm:** each 1 node/task, 20 CPUs, high-mem, 256 GiB, 23 h.
- **Status:** **Structurally complete; execution unverified; expensive.** Three near-duplicate sources invite drift.
- **First clean workflow:** **Conditional.** Valuable for MPS analysis but ordering/normalization and cost require approval.

### A8. Legacy Python Schmidt/MPS construction library and notebook

- **Purpose:** Explore Schmidt decompositions, site permutations, critical bond dimensions, entropy, Python reconstruction, Tucker comparisons, spectra, and ad hoc visualization across channel-flow and forced-isotropic cases.
- **Entry scripts and local imports:** interactive [`Schmidt_decomp.ipynb`](../Schmidt_decomp.ipynb); callable roots [`mps_py_codes/process_file.py:20`](../mps_py_codes/process_file.py#L20) and [`mps_py_codes/process_file_1024.py:19`](../mps_py_codes/process_file_1024.py#L19). Their local dependency chain is [`velocity_name.py`](../mps_py_codes/velocity_name.py), [`reshape_and_transpose.py`](../mps_py_codes/reshape_and_transpose.py) -> [`MPS_permute.py`](../mps_py_codes/MPS_permute.py), [`prime_sort.py`](../mps_py_codes/prime_sort.py), [`FindSingValsAndEntEntr.py`](../mps_py_codes/FindSingValsAndEntEntr.py), [`CritChis_uneq.py`](../mps_py_codes/CritChis_uneq.py), [`comp_ratio_uneq.py`](../mps_py_codes/comp_ratio_uneq.py), [`calculate_u_comp_fidelity_l2norm.py`](../mps_py_codes/calculate_u_comp_fidelity_l2norm.py), and [`save_as_mat.py`](../mps_py_codes/save_as_mat.py); the import roots are evidenced at [`mps_py_codes/process_file.py:11`](../mps_py_codes/process_file.py#L11) and [`mps_py_codes/process_file_1024.py:10`](../mps_py_codes/process_file_1024.py#L10).
- **Language/environment:** notebook kernel `myenv`, Python 3.11.4; NumPy, SciPy, h5py, pandas, matplotlib, tikzplotlib, tensorly, PIL, and local helpers. No environment lock.
- **Inputs:** numerous relative and absolute `.mat`, `.h5`, binary, CSV, and serialized examples. Central path registry is [`mps_py_codes/config_FI.json`](../mps_py_codes/config_FI.json), containing CRC, macOS, Windows placeholder, and relative paths.
- **Outputs/metrics:** reconstructed fields, singular values/entropy, CR/fidelity/L2, flow metrics, plots, and notebook displays. The notebook has 216 cells, 81 executed code cells, and seven stored error outputs.
- **Slurm:** none for the notebook/library as a whole.
- **Status:** **Experimental/incomplete as a reproducible workflow.** It is a valuable source of algorithms, not one deterministic entry path.
- **First clean workflow:** **No.** Use only after choosing a specific historical algorithm/case; do not treat the notebook as authority.

### A9. Legacy forced-isotropic MPS quality and flow-statistics suite

- **Purpose:** Evaluate JHS DNS vs MPS reconstruction: CR/fidelity/L2; flow scalars and spectrum; contour/scatter; gradient PDFs; velocity and dissipation PDFs; strain-rate-eigenvalue PDFs; Q-R joint PDFs; and compressed-only spectrum.
- **Entry scripts and local imports:**
  - quality: [`py/1FI_MPS_3D_1024cubed_t1_cr.py:1`](../py/1FI_MPS_3D_1024cubed_t1_cr.py#L1) -> `calculate_u_comp_fidelity_l2norm`, `comp_ratio_uneq`, `process_file_1024`, `save_as_mat`;
  - flow/scalars: [`py/2FI_MPS_3D_1024cubed_t1_scalars.py:1`](../py/2FI_MPS_3D_1024cubed_t1_scalars.py#L1) -> `process_file_1024`, `calculate_flow_parameters_comb`, `save_as_mat`;
  - plots/derived quantities: [`py/3FI_MPS_3D_1024cubed_t1_scatter.py:1`](../py/3FI_MPS_3D_1024cubed_t1_scatter.py#L1), [`py/4FI_MPS_3D_1024cubed_t1_pdf_grad.py:1`](../py/4FI_MPS_3D_1024cubed_t1_pdf_grad.py#L1), [`py/5FI_MPS_3D_1024cubed_t1_pdf_u_disp.py:1`](../py/5FI_MPS_3D_1024cubed_t1_pdf_u_disp.py#L1), [`py/6FI_MPS_3D_1024cubed_t1_stilde.py:1`](../py/6FI_MPS_3D_1024cubed_t1_stilde.py#L1), [`py/7FI_MPS_3D_1024cubed_t1_joint_pdf_QR.py:1`](../py/7FI_MPS_3D_1024cubed_t1_joint_pdf_QR.py#L1), [`py/8FI_MPS_3D_1024cubed_t1_spectrum.py:1`](../py/8FI_MPS_3D_1024cubed_t1_spectrum.py#L1). Local imports are [`plotting.py`](../mps_py_codes/plotting.py), [`calculate_flow_parameters_comb.py`](../mps_py_codes/calculate_flow_parameters_comb.py), [`calculate_orientation.py`](../mps_py_codes/calculate_orientation.py), [`calculate_eps_stilde.py`](../mps_py_codes/calculate_eps_stilde.py), [`calculate_gradients.py`](../mps_py_codes/calculate_gradients.py), [`calculate_spectrum.py`](../mps_py_codes/calculate_spectrum.py), [`process_file_1024.py`](../mps_py_codes/process_file_1024.py), and [`save_as_mat.py`](../mps_py_codes/save_as_mat.py).
- **Language/environment:** Python 3.11 module; NumPy/SciPy/h5py/matplotlib/tikzplotlib/cmasher plus local code. Most entries append `/ix/pgivi/moe32/Schmidt/mps_py_codes`, while two append a different `/ix/.../Nik_code/.../mps_py_codes`, rather than this checkout ([`py/1FI_MPS_3D_1024cubed_t1_cr.py:3`](../py/1FI_MPS_3D_1024cubed_t1_cr.py#L3), [`py/2FI_MPS_3D_1024cubed_t1_scalars.py:1`](../py/2FI_MPS_3D_1024cubed_t1_scalars.py#L1)).
- **Inputs:** hard-coded `cutoff=1e-2`, local reconstructed MATs, JHS paths from `config_FI.json`, and precomputed HDF5 gradients/dissipation/invariants/eigenvalues. Input naming alternates uppercase/lowercase components.
- **Outputs/metrics:** MAT flow/CR/fidelity/L2/spectrum records; PDF/PNG/TEX contours, scatters, PDFs, and Q-R plots; intermediate HDF5 derivatives/dissipation are written by `calculate_flow_parameters_comb` ([`mps_py_codes/calculate_flow_parameters_comb.py:64`](../mps_py_codes/calculate_flow_parameters_comb.py#L64)).
- **Slurm:** one 23 h high-memory wrapper per numbered script: [`11FI_MPS_3D_1024cubed_t1_cr.slurm`](../Slurms/11FI_MPS_3D_1024cubed_t1_cr.slurm), [`22FI_MPS_3D_1024cubed_t1_scalars.slurm`](../Slurms/22FI_MPS_3D_1024cubed_t1_scalars.slurm), [`33FI_MPS_3D_1024cubed_t1_scatter.slurm`](../Slurms/33FI_MPS_3D_1024cubed_t1_scatter.slurm), [`44FI_MPS_3D_1024cubed_t1_pdf_grad.slurm`](../Slurms/44FI_MPS_3D_1024cubed_t1_pdf_grad.slurm), [`55FI_MPS_3D_1024cubed_t1_pdf_u_disp.slurm`](../Slurms/55FI_MPS_3D_1024cubed_t1_pdf_u_disp.slurm), [`66FI_MPS_3D_1024cubed_t1_stilde.slurm`](../Slurms/66FI_MPS_3D_1024cubed_t1_stilde.slurm), [`77FI_MPS_3D_1024cubed_t1_joint_pdf_QR.slurm`](../Slurms/77FI_MPS_3D_1024cubed_t1_joint_pdf_QR.slurm), and [`88FI_MPS_3D_1024cubed_t1_spectrum.slurm`](../Slurms/88FI_MPS_3D_1024cubed_t1_spectrum.slurm); requests range from 256 to 512 GiB.
- **Status:** **Incomplete/non-portable in this checkout and scientifically uncertain.** Hard-coded external imports, working-directory inputs, active/commented alternatives, and precomputed intermediates prevent a single evidenced path.
- **First clean workflow:** **No.** The newer stats/spectrum paths should establish conventions before salvaging individual legacy analyses.

### A10. Legacy forced-isotropic structure functions

- **Purpose:** Compute checkpointed two-point correlations and second/third-order structure functions for DNS velocity along x, y, or z; commented code can instead process an MPS reconstruction.
- **Entry scripts and local imports:** [`py/81FI_MPS_3D_1024cubed_t1_structure_func_x.py:1`](../py/81FI_MPS_3D_1024cubed_t1_structure_func_x.py#L1), [`py/82FI_MPS_3D_1024cubed_t1_structure_func_y.py:1`](../py/82FI_MPS_3D_1024cubed_t1_structure_func_y.py#L1), and [`py/83FI_MPS_3D_1024cubed_t1_structure_func_z.py:1`](../py/83FI_MPS_3D_1024cubed_t1_structure_func_z.py#L1) -> [`process_file_1024.py`](../mps_py_codes/process_file_1024.py), [`structure_functions.safe_struct`](../mps_py_codes/structure_functions.py#L582), and [`save_as_mat.py`](../mps_py_codes/save_as_mat.py).
- **Language/environment:** same legacy Python environment as A9.
- **Inputs:** JHS `{U,V,W}_t1.mat` paths from `config_FI.json`; active path computes DNS only. Checkpoints are `lag_<direction>_<stop>.npz` in a working-directory checkpoint folder ([`mps_py_codes/structure_functions.py:624`](../mps_py_codes/structure_functions.py#L624)).
- **Outputs/metrics:** `sf_DNS_dir_<axis>_comp.mat` plus resumable `.npz` checkpoints; structure-function arrays.
- **Slurm:** three high-mem wrappers, each 256 GiB and 71 h: [`Slurms/881FI_MPS_3D_1024cubed_t1_structure_func_x.slurm:1`](../Slurms/881FI_MPS_3D_1024cubed_t1_structure_func_x.slurm#L1), [`Slurms/882FI_MPS_3D_1024cubed_t1_structure_func_y.slurm:1`](../Slurms/882FI_MPS_3D_1024cubed_t1_structure_func_y.slurm#L1), and [`Slurms/883FI_MPS_3D_1024cubed_t1_structure_func_z.slurm:1`](../Slurms/883FI_MPS_3D_1024cubed_t1_structure_func_z.slurm#L1).
- **Status:** **Structurally present; execution unverified; very expensive.** MPS branch is commented and filenames still say `_comp` for DNS.
- **First clean workflow:** **No.** Cost and ambiguous active/compressed intent make it a later target.

## B. LES, filtered-DNS, and velocity-spectrum workflows

### B1. CRC block filter with local web job manager

- **Purpose:** Downsample a cubic scalar/velocity field by non-overlapping `delta^3` block means, compute variance/SGS-stress closure checks, save the filtered cube, and submit/monitor CRC jobs through a browser UI.
- **Entry scripts and local imports:** CRC compute [`LES_interface/crc/run_les.jl:1`](../LES_interface/crc/run_les.jl#L1) -> `include("LES_filter.jl")`; filter library actually checked in at [`LES_interface/LES_filter.jl:3`](../LES_interface/LES_filter.jl#L3); Slurm template [`LES_interface/crc/les_filter.slurm.template:1`](../LES_interface/crc/les_filter.slurm.template#L1); Python/SSH UI [`LES_interface/crc/server.py:1`](../LES_interface/crc/server.py#L1) and launcher [`LES_interface/crc/start_les.command:1`](../LES_interface/crc/start_les.command#L1). An earlier local Julia server is [`LES_interface/server.jl:1`](../LES_interface/server.jl#L1) -> `LES_interface/LES_filter.jl`, launched by [`LES_interface/start_les_filter.command:1`](../LES_interface/start_les_filter.command#L1).
- **Language/environment:** CRC Julia with `Dates` plus the filter's `MAT` dependency inherited from deployment; version/package project is unpinned. Python 3 standard library manages SSH/SCP/browser; local Julia server needs HTTP, JSON3, MAT, Dates.
- **Inputs:** MAT (first dataset) or native `Float64` binary with explicit cube `N` and byte skip; CLI documented at [`LES_interface/crc/run_les.jl:27`](../LES_interface/crc/run_les.jl#L27). UI maps `.bin` names to grid/header defaults in [`LES_interface/crc/server.py:233`](../LES_interface/crc/server.py#L233).
- **Outputs/metrics:** `filtered_<name>_<M>x<M>x<M>.mat` key `u`; report/status JSON; mean preservation, total/resolved variance, mean SGS stress, residual, resolved ratio, load/filter/SGS/total timings ([`LES_interface/crc/run_les.jl:118`](../LES_interface/crc/run_les.jl#L118), [`LES_interface/crc/run_les.jl:135`](../LES_interface/crc/run_les.jl#L135)). UI records history and active jobs.
- **Slurm:** templated CPUs, memory, and time; 1 node/task, SMP. The template runs Julia with `$SLURM_CPUS_PER_TASK` and organizes results into a per-job directory ([`LES_interface/crc/les_filter.slurm.template:2`](../LES_interface/crc/les_filter.slurm.template#L2)).
- **Status:** **Known working externally, incomplete as checked out.** `les_history.json` records 29 completed CRC jobs, including 1024³ JHS factors 8/16/32/64. However `run_les.jl` expects `crc/LES_filter.jl`, which is absent; the library is one directory up. The UI also targets external `/ix/pgivi/moe32/LES_filter` ([`LES_interface/crc/server.py:23`](../LES_interface/crc/server.py#L23)).
- **First clean workflow:** **Yes, after a routine packaging repair and explicit confirmation that the recorded block-average definition is the desired reference.** It has the strongest run evidence and measurable invariants.

### B2. Early same-grid neighborhood filter experiment

- **Purpose:** Apply a separable periodic moving box filter that retains original grid size, then compare total, resolved, and SGS variance.
- **Entry scripts and local imports:** [`Julia/LES_filter_1024_main.jl:1`](../Julia/LES_filter_1024_main.jl#L1) -> [`Julia/LES_filter.jl:1`](../Julia/LES_filter.jl#L1).
- **Language/environment:** Julia, MAT, LinearAlgebra, Statistics, Plots; version/project unrecorded.
- **Inputs:** hard-coded JHS `U_t1.mat`; radius `(1,1,1)` ([`Julia/LES_filter_1024_main.jl:6`](../Julia/LES_filter_1024_main.jl#L6)).
- **Outputs/metrics:** stdout relative norm and variance decomposition. MAT writes are commented.
- **Slurm:** none.
- **Status:** **Experimental/incomplete.** This is a different filtering operator and output grid than B1; the catalog does not choose between them.
- **First clean workflow:** **No.** Only standardize if this filter, rather than block coarsening, is explicitly selected.

### B3. Ideal-LES filtered DNS cube and flame statistics

- **Purpose:** Block-average jet DNS `512^3` cubes to a coarser ideal-LES reference, cache them as NPY, reconstruct chi on the coarse grid, and run the shared flame-statistics schema.
- **Entry scripts and local imports:** filter/cache [`py/make_filtered_cube.py:66`](../py/make_filtered_cube.py#L66) -> `plot_qc4pde._parse`; stats [`py/run_filtered_stats.py:80`](../py/run_filtered_stats.py#L80) -> `dns_stats`, `derived_fields`; related scalar/velocity energy reports [`py/mixfrac_energy_ratio.py:71`](../py/mixfrac_energy_ratio.py#L71) and [`py/ke_energy_ratio.py:74`](../py/ke_energy_ratio.py#L74) -> runtime imports from `dns_stats`.
- **Language/environment:** Python/NumPy; stats also use pickle and modern local modules. No dedicated Slurm wrapper is checked in.
- **Inputs:** DNS `jet_<field>_<ts>.dat`; default factor 8; cached `filtered_<field>_<ts>_f<factor>.npy` ([`py/make_filtered_cube.py:19`](../py/make_filtered_cube.py#L19)). Stats require mixfrac, T, Y_O2, Y_OH, Y_CO2 and optionally Y_CO/alpha ([`py/run_filtered_stats.py:103`](../py/run_filtered_stats.py#L103)).
- **Outputs/metrics:** NPY cubes; `filtered_stats_<ts>.pkl`; standard profiles/PDFs/conditional/joint/manifold/extinction stats; corrected coarse y-axis; text ratios `<field^2>`/variance or combined kinetic-energy ratios ([`py/run_filtered_stats.py:200`](../py/run_filtered_stats.py#L200), [`py/ke_energy_ratio.py:147`](../py/ke_energy_ratio.py#L147)).
- **Slurm:** none.
- **Status:** **Structurally complete; execution unverified.** `make_filtered_cube` defaults to field token `temp`, while the shared stats require cached `T`; a caller must choose the evidenced file token appropriate to the dataset. The chi formula/operator is also a scientific convention.
- **First clean workflow:** **Conditional.** Low-to-moderate cost and broadly useful, but filter factor, field naming, chi construction, and boundary behavior need confirmation.

### B4. AMReX/PeleLM LES-FDF statistics

- **Purpose:** Load an AMReX plotfile with yt, extract the centered `64^3` physical counterpart of the DNS cube, select stored/reconstructed/FDF chi, and compute the same statistics schema as DNS/MPS/PEPS.
- **Entry scripts and local imports:** [`py/run_les_stats.py:49`](../py/run_les_stats.py#L49) -> `dns_stats`, `derived_fields`, `les_io`; loader [`py/les_io.py:87`](../py/les_io.py#L87); wrapper [`sh/submit_les_stats.sh:1`](../sh/submit_les_stats.sh#L1).
- **Language/environment:** Python 3.11, `my_env`, NumPy, yt, pickle; optional plotting dependencies are loaded by wrapper environment.
- **Inputs:** default `jet_<ts>/LES_plt20000`; yt field aliases are enumerated in [`py/les_io.py:50`](../py/les_io.py#L50). FDF mode also reads `Tauplt<N>.temp` field `Freq` and `TauAllplt<N>.temp` field `Z`; auto-detection is at [`py/run_les_stats.py:73`](../py/run_les_stats.py#L73).
- **Outputs/metrics:** `les_stats_<ts>...pkl` with standard flame statistics and metadata; stored/mixed/reconstructed/FDF chi diagnostics.
- **Slurm:** 4 CPUs, 32 GiB, 1 h, SMP ([`sh/submit_les_stats.sh:2`](../sh/submit_les_stats.sh#L2)).
- **Status:** **Structurally complete; execution unverified; scientifically conditional.** Missing stored chi may be replaced with zeros or reconstruction depending on mode ([`py/run_les_stats.py:100`](../py/run_les_stats.py#L100)); axis/orientation has a separate diagnostic rather than retained resolution.
- **First clean workflow:** **Conditional.** Valuable after LES field mapping, axis convention, and chi mode are approved.

### B5. Velocity DNS/MPS/ideal-LES spectrum and statistics

- **Purpose:** Compare three velocity representations via energy spectrum, contours, PDFs, mean/RMS, and resolved energy ratios. There are separate jet `512^3` and JHS `1024^3` implementations.
- **Entry scripts and local imports:** jet [`py/velocity_spectrum_stats.py:205`](../py/velocity_spectrum_stats.py#L205), no local imports; JHS [`py/jhs_velocity_spectrum_stats.py:245`](../py/jhs_velocity_spectrum_stats.py#L245), no local imports; JHS wrapper [`sh/submit_jhs_analysis.sh:1`](../sh/submit_jhs_analysis.sh#L1).
- **Language/environment:** Python/NumPy/matplotlib; h5py (and fallback SciPy) for JHS. Jet loader imports `dns_stats` at runtime. JHS comments document a NumPy/SciPy ABI problem and prefer h5py ([`py/jhs_velocity_spectrum_stats.py:17`](../py/jhs_velocity_spectrum_stats.py#L17)).
- **Inputs:** jet big-endian `.dat` `jet_{u,v,w}_<ts>.dat`, MPS reconstruction pattern, cached filtered NPY; JHS `{U,V,W}_t1.mat`, `truncated_{component}_t1_1024_il_chi<N>.mat`, filtered NPY ([`py/velocity_spectrum_stats.py:13`](../py/velocity_spectrum_stats.py#L13), [`py/jhs_velocity_spectrum_stats.py:32`](../py/jhs_velocity_spectrum_stats.py#L32)).
- **Outputs/metrics:** spectrum PNG/PDF (and JHS MAT), contour PNG/PDF, velocity PDF PNG/PDF, text mean/RMS and kinetic-energy ratios ([`py/jhs_velocity_spectrum_stats.py:10`](../py/jhs_velocity_spectrum_stats.py#L10)).
- **Slurm:** JHS wrapper requests high-mem, 8 CPUs, 480 GiB, 6 h; one `(chi,factor)` case per submission ([`sh/submit_jhs_analysis.sh:2`](../sh/submit_jhs_analysis.sh#L2)). Jet has no wrapper.
- **Status:** **Structurally complete; execution unverified; high cost.** JHS docstring warns peak memory well above 100 GiB ([`py/jhs_velocity_spectrum_stats.py:23`](../py/jhs_velocity_spectrum_stats.py#L23)). Spectrum formula/normalization and MPS pattern require confirmation.
- **First clean workflow:** **No for computation; conditional for plotting cached spectra.**

## C. Physical statistics, validation, and reporting workflows

### C1. Jet DNS flame statistics, derived-field validation, and DNS plots

- **Purpose:** Load one DNS snapshot (center cube, full grid, or both), reproduce the paper's profiles/PDFs/conditional statistics/extinction metrics, validate Bilger Z and scalar-dissipation reconstruction, sweep finite-difference settings, and render DNS figures.
- **Entry scripts and local imports:** compute [`py/run_dns_stats.py:54`](../py/run_dns_stats.py#L54) -> `dns_stats`, `derived_fields`; plotting [`py/plot_dns_stats.py:612`](../py/plot_dns_stats.py#L612); dedicated chi sweep [`py/sweep_dns_chi_recon.py:109`](../py/sweep_dns_chi_recon.py#L109) -> `dns_stats`, `derived_fields`; replot [`py/replot_chi_sweep.py:90`](../py/replot_chi_sweep.py#L90); wrappers [`sh/submit_dns_stats.sh:1`](../sh/submit_dns_stats.sh#L1) and [`sh/submit_dns_chi_sweep.sh:1`](../sh/submit_dns_chi_sweep.sh#L1).
- **Language/environment:** Python 3.11, `my_env`, NumPy, matplotlib, pickle; wrapper also loads texlive. No pinned versions.
- **Inputs:** `jet_<field>_<timestep>.dat`, big-endian float32 full grid. Required core fields are mixfrac, T, Y_CO, Y_CO2, Y_OH, Y_O2, chi; optional chemistry/transport/velocity fields are loaded when present ([`py/dns_stats.py:180`](../py/dns_stats.py#L180)). Default wrapper uses timestep 0198 although README prose mentions 0228.
- **Outputs/metrics:** per-region `dns_stats_<ts>.pkl`; validation pickles; paper-style PNGs and `summary.txt`; mean/RMS, thickness, chi profile, conditional T, extinction, marginal/joint PDFs, manifold sample, Z/chi reconstruction errors, and FD-order/periodicity diagnostics ([`py/run_dns_stats.py:94`](../py/run_dns_stats.py#L94), [`py/plot_dns_stats.py:78`](../py/plot_dns_stats.py#L78)). The driver docstring says NPZ but code writes pickle.
- **Slurm:** main wrapper 4 CPUs, 180 GiB, 2 h, SMP and defaults to both cube/full plus all validation ([`sh/submit_dns_stats.sh:2`](../sh/submit_dns_stats.sh#L2)); dedicated chi sweep 4 CPUs, 64 GiB, 5 h.
- **Status:** **Structurally complete; execution unverified.** README documents this as a pipeline ([`README_JetFlame.md:1`](../README_JetFlame.md#L1)), and synthetic tests exist, but no run artifacts are retained. Scientific constants, formulas, region, timestep, and operator remain choices for standardization.
- **First clean workflow:** **Yes, strongest end-to-end scientific candidate once one dataset/timestep and existing conventions are explicitly accepted.** It is the reference schema for nearly every later statistics comparison.

### C2. Jet MPS flame statistics and cutoff/order campaign

- **Purpose:** Load one MPS field set, choose stored/mixed/reconstructed Z/chi handling, compute the C1 statistics schema, sweep cutoff/order/operator cases, and trigger DNS-vs-MPS comparison plots.
- **Entry scripts and local imports:** [`py/run_mps_stats.py:55`](../py/run_mps_stats.py#L55) -> `dns_stats`, `mps_io`, `derived_fields`; wrappers [`sh/submit_mps_stats.sh:1`](../sh/submit_mps_stats.sh#L1), [`sh/submit_mps_stats_cf4.1e-4.sh:1`](../sh/submit_mps_stats_cf4.1e-4.sh#L1), [`sh/submit_mps_stats_chi93.sh:1`](../sh/submit_mps_stats_chi93.sh#L1); campaign submitter [`sh/submit_all_cases.sh:1`](../sh/submit_all_cases.sh#L1).
- **Language/environment:** modern Python stack: NumPy, h5py, pickle, local stats modules, `my_env`, Python 3.11, texlive.
- **Inputs:** MPS MAT family documented under repository-wide findings; required seven fields for stored mode, with alpha/species needed by mixed/recon. Wrapper searches `truncated_<ts>/sp<N>`, legacy flat, then no-sp layouts ([`sh/submit_mps_stats.sh:42`](../sh/submit_mps_stats.sh#L42)).
- **Outputs/metrics:** `mps_stats_<ts>_cf<label>_<mode>...pkl`, standard flame statistics and reconstruction metadata; campaign comparison plots through C4 ([`py/run_mps_stats.py:170`](../py/run_mps_stats.py#L170)).
- **Slurm:** four-task cutoff array, 4 CPUs/task, 64 GiB, 1 h; dependent comparison job 2 CPUs, 32 GiB, 45 min ([`sh/submit_mps_stats.sh:2`](../sh/submit_mps_stats.sh#L2), [`sh/submit_mps_stats.sh:115`](../sh/submit_mps_stats.sh#L115)). Full campaign can submit many arrays across three chi modes, two FD orders, two periodicity values, and four orderings.
- **Status:** **Structurally complete; execution unverified; scientifically unresolved.** Its three chi modes intentionally encode different physical questions ([`py/run_mps_stats.py:12`](../py/run_mps_stats.py#L12)); the catalog does not select one.
- **First clean workflow:** **Conditional.** High reuse/value, but should follow C1 and A1 decisions; do not standardize the full campaign first.

### C3. PEPS flame statistics and inspection

- **Purpose:** Load PEPS D=9 fields, optionally sign/L2-rescale each against DNS, reconstruct chi, compute the shared flame statistics, and inspect ranges/orientation.
- **Entry scripts and local imports:** [`py/run_peps_stats.py:49`](../py/run_peps_stats.py#L49) -> `dns_stats`, `derived_fields`, `peps_io`; wrapper [`sh/submit_peps_stats.sh:1`](../sh/submit_peps_stats.sh#L1); pickle inspection [`sh/submit_peps_inspect.sh:1`](../sh/submit_peps_inspect.sh#L1); visual orientation check [`sh/submit_peps_orient_check.sh:1`](../sh/submit_peps_orient_check.sh#L1) -> inline imports `dns_stats`, `peps_io`.
- **Language/environment:** modern Python stack plus h5py; 3.11 module, `my_env`, optional texlive.
- **Inputs:** `truncated_<ts>/PEPS/<var>_wf_D=9_periodic.mat`; required mixfrac/T, optional alpha and four species; dataset key inferred by field ([`py/peps_io.py:27`](../py/peps_io.py#L27)). DNS `.dat` inputs are used for default scale/sign correction.
- **Outputs/metrics:** `peps_stats_<ts>_<tag>_<mode>...pkl`; standard stats, rescaling metadata; inspection stdout; `QC4PDE/peps_orient_check.png`.
- **Slurm:** stats 4 CPUs, 64 GiB, 1 h; inspect 1 CPU, 4 GiB, 5 min; orientation 2 CPUs, 16 GiB, 10 min.
- **Status:** **Structurally complete; experimental/uncertain scientifically.** Loader docstring reports empirical transpose correlation, while orientation is still delegated to a visual check ([`py/peps_io.py:55`](../py/peps_io.py#L55)). Sign/L2 normalization is a major scientific choice.
- **First clean workflow:** **No until PEPS orientation and normalization are approved.**

### C4. Cross-source and cross-order statistics comparison

- **Purpose:** Consume existing DNS/MPS/LES/PEPS/filtered statistics pickles and produce overlays, per-source joint PDFs/manifolds, error summaries, and cross-order (`sp`) comparisons.
- **Entry scripts and local imports:** [`py/plot_dns_mps_comparison.py:781`](../py/plot_dns_mps_comparison.py#L781) -> `save_tikz`; [`py/plot_sp_overlay.py:271`](../py/plot_sp_overlay.py#L271); automatically invoked by [`sh/submit_mps_stats.sh:98`](../sh/submit_mps_stats.sh#L98).
- **Language/environment:** Python/NumPy/matplotlib/pickle; optional tikzplotlib and LaTeX.
- **Inputs:** result tree `dns_<ts>/cube`, `mps_<ts>_sp<N>/<cutoff>`, optionally `les_<ts>`, `peps_<ts>`, and filtered results. Explicit five-pickle mode is also supported ([`py/plot_dns_mps_comparison.py:7`](../py/plot_dns_mps_comparison.py#L7)).
- **Outputs/metrics:** PNG/TEX overlays for mean/RMS, chi, conditional T, and Z PDF; per-source joint PDF and manifold PNGs; `comparison_summary.txt`; cross-sp PNGs for four statistic families ([`py/plot_dns_mps_comparison.py:559`](../py/plot_dns_mps_comparison.py#L559), [`py/plot_sp_overlay.py:11`](../py/plot_sp_overlay.py#L11)).
- **Slurm:** dependent job: 2 CPUs, 32 GiB, 45 min; no standalone static wrapper for cross-sp.
- **Status:** **Structurally complete; execution unverified.** Low compute once pickles exist, but semantic comparability depends on upstream mode/operator/axis choices.
- **First clean workflow:** **Yes as a safe third candidate after C1 outputs exist.** It offers high reuse and catches cross-source schema drift without rebuilding tensors.

### C5. Field-level DNS/MPS scatter, PDF, and contour campaign

- **Purpose:** Compare individual mixfrac/T/chi fields across DNS and a 4x4 order/cutoff grid using pointwise scatter metrics, PDFs, and mid-plane contours; aggregate scatter metrics to CSV/XLSX.
- **Entry scripts and local imports:** [`py/plot_field_scatter.py:180`](../py/plot_field_scatter.py#L180), [`py/plot_field_pdf.py:139`](../py/plot_field_pdf.py#L139), [`py/plot_field_contour.py:131`](../py/plot_field_contour.py#L131) -> `dns_stats`, `derived_fields`, `mps_io`; aggregator [`py/aggregate_scatter_metrics.py:250`](../py/aggregate_scatter_metrics.py#L250); submitter [`sh/submit_field_plots.sh:1`](../sh/submit_field_plots.sh#L1).
- **Language/environment:** modern Python/NumPy/h5py/matplotlib/pandas/Excel writer; Python 3.11 module and `my_env`.
- **Inputs:** DNS `.dat`, MPS MAT grid, optional LES AMReX plotfile; user-selectable field, order, cutoff, chi mode, FD order, and periodicity.
- **Outputs/metrics:** scatter PNG+JSON (`rms`, relative L2, correlation), per-field CSV/XLSX grids, eight PDF overlays per field, and up to 34 contour plots per field ([`py/plot_field_scatter.py:251`](../py/plot_field_scatter.py#L251), [`py/plot_field_pdf.py:4`](../py/plot_field_pdf.py#L4), [`py/plot_field_contour.py:3`](../py/plot_field_contour.py#L3)).
- **Slurm:** dynamic submitter defaults to 48 scatter jobs (2 CPUs, 24 GiB, 30 min), three PDF jobs and three contour jobs (4 CPUs, 64 GiB, 2 h), plus a 1 CPU/4 GiB aggregator ([`sh/submit_field_plots.sh:1`](../sh/submit_field_plots.sh#L1)).
- **Status:** **Structurally complete; execution unverified; potentially expansive.** Submission count is material and requires explicit approval.
- **First clean workflow:** **No as a campaign; conditional for one read-only plot case.**

### C6. QC4PDE paper figures and Z metrics

- **Purpose:** Assemble paper-ready multi-source figures and report Z-field fidelity/infidelity and profile errors.
- **Entry scripts and local imports:** figures [`py/plot_qc4pde.py:718`](../py/plot_qc4pde.py#L718) -> `save_tikz`; metrics [`py/compute_qc4pde_metrics.py:277`](../py/compute_qc4pde_metrics.py#L277) with runtime imports `dns_stats`, `mps_io`, `peps_io`; wrappers [`sh/submit_qc4pde.sh:1`](../sh/submit_qc4pde.sh#L1), [`sh/submit_qc4pde_metrics.sh:1`](../sh/submit_qc4pde_metrics.sh#L1).
- **Language/environment:** modern Python/NumPy/matplotlib/pickle/h5py, optional tikzplotlib/LaTeX; `my_env`, Python 3.11.
- **Inputs:** C1-C3/B4 stats tree plus raw DNS/MPS/PEPS cubes and optional LES plotfile/filtered stats. Default source order is DNS, LES, MPS chi93, MPS 4.1e-4, PEPS ([`py/plot_qc4pde.py:11`](../py/plot_qc4pde.py#L11)).
- **Outputs/metrics:** PNG/PDF/TEX contours, Z mean/RMS, Z PDF, conditional T, joint PDF, manifold; `metrics_Z.txt` with field fidelity/infidelity and profile relative L2/RMS/peak errors ([`py/compute_qc4pde_metrics.py:4`](../py/compute_qc4pde_metrics.py#L4)).
- **Slurm:** each wrapper 2 CPUs, 32 GiB, 30 min, SMP.
- **Status:** **Structurally complete; execution unverified; publication-specific.** PEPS field metrics apply sign/L2 correction, LES field infidelity is skipped for grid mismatch.
- **First clean workflow:** **Conditional.** Safe reporting target once upstream source preparation is fixed; not a first scientific compute path.

### C7. Gradient, scalar-dissipation, and reconstruction diagnostics

- **Purpose:** Diagnose FD/operator sensitivity, conditional gradient amplification, MPS seam spikes, full-field reconstruction errors, and the Z/gradient/alpha decomposition of chi.
- **Entry scripts and local imports:**
  - DNS FD sweep: [`py/sweep_dns_chi_recon.py:109`](../py/sweep_dns_chi_recon.py#L109) -> `dns_stats`, `derived_fields`; wrapper `submit_dns_chi_sweep.sh`;
  - conditional gradients: [`py/plot_gradient_stats.py:221`](../py/plot_gradient_stats.py#L221) -> `plot_qc4pde`;
  - seams/spikes: [`py/diagnose_mps_spikes.py:145`](../py/diagnose_mps_spikes.py#L145) -> `plot_qc4pde`, `plot_gradient_stats`; wrapper [`sh/submit_spike_sweep.sh:1`](../sh/submit_spike_sweep.sh#L1);
  - chi decomposition: [`py/plot_chi_decomposition.py:308`](../py/plot_chi_decomposition.py#L308) -> `dns_stats`, `derived_fields`, `mps_io`, `save_tikz`; wrapper [`sh/submit_chi_decomposition.sh:1`](../sh/submit_chi_decomposition.sh#L1);
  - full-field Z errors: [`py/field_l2_infidelity.py:85`](../py/field_l2_infidelity.py#L85) -> `plot_gradient_stats`.
- **Language/environment:** modern Python/NumPy/matplotlib/h5py/pickle and optional tikz/LaTeX.
- **Inputs:** raw DNS Z/alpha/rho, MPS `chi93`/cutoff cases, PEPS D=9, and/or saved DNS sweep pickle. Exact defaults are timestep 0198, sp1, order 4 or 8 depending on entry.
- **Outputs/metrics:** sweep PKL/TXT/PNGs; `cond_gradZ`/`cond_gradZ2` PNG/PDF; seam diagnostic PNG and stdout; chi-decomposition profile/PDF/scatter PNG/TEX plus text metrics; `field_l2_infidelity.txt` ([`py/field_l2_infidelity.py:136`](../py/field_l2_infidelity.py#L136)).
- **Slurm:** chi sweep 64 GiB/5 h; decomposition 64 GiB/1 h; spike sweep 96 GiB/2 h. Gradient and field-L2 entries have no wrappers.
- **Status:** **Experimental diagnostics; execution unverified.** They intentionally explore disputed boundary/operator, seam, normalization, and fidelity choices rather than establish them.
- **First clean workflow:** **No.** Use to inform later decisions after a baseline is selected.

### C8. LES/PEPS orientation diagnostics

- **Purpose:** Test whether physical planes and axis orientations align between DNS and LES/PEPS representations.
- **Entry scripts and local imports:** LES exhaustive correlation diagnostic [`py/diagnose_les_plane.py:156`](../py/diagnose_les_plane.py#L156), runtime imports `dns_stats`, `les_io`; wrapper [`sh/submit_les_plane_diag.sh:1`](../sh/submit_les_plane_diag.sh#L1). PEPS visual panel is inline in [`sh/submit_peps_orient_check.sh:1`](../sh/submit_peps_orient_check.sh#L1), importing `dns_stats` and `peps_io`.
- **Language/environment:** modern Python/NumPy/matplotlib; yt for LES, h5py for PEPS.
- **Inputs:** DNS mixfrac `.dat`, LES AMReX plotfile, or PEPS mixfrac MAT.
- **Outputs/metrics:** LES best permutation/flip/slice correlation report and PNG; PEPS 3x3 visual orientation grid.
- **Slurm:** LES 4 CPUs, 64 GiB, 45 min; PEPS 2 CPUs, 16 GiB, 10 min.
- **Status:** **Experimental and unresolved.** The presence of these scripts is evidence that orientation was under investigation, not that one orientation is authoritative.
- **First clean workflow:** **No as a production workflow; yes as a prerequisite diagnostic if orientation evidence is missing.**

### C9. DNS box-face visualization

- **Purpose:** Render two real-data faces of the DNS mixture-fraction cube in a publication-style 3D schematic.
- **Entry scripts and local imports:** [`py/plot_box_faces.py:105`](../py/plot_box_faces.py#L105); runtime import `dns_stats`; no wrapper.
- **Language/environment:** Python/NumPy/matplotlib, optional system LaTeX.
- **Inputs:** `jet_mixfrac_<ts>.dat` and `GridConfig` coordinates.
- **Outputs/metrics:** transparent `box_faces.png`; visualization only ([`py/plot_box_faces.py:282`](../py/plot_box_faces.py#L282)).
- **Slurm:** none.
- **Status:** **Structurally complete; execution unverified.**
- **First clean workflow:** **No.** Low risk but too narrow to establish shared CRC/data conventions.

### C10. Synthetic validation and reference-contract inspection

- **Purpose:** Exercise the shared DNS statistics on a tiny synthetic binary dataset; test Bilger stream conversion and analytical chi gradients; inspect the provisional Julia MPS source text without evaluating it; capture a future Julia/Slurm environment.
- **Entry scripts and local imports:** [`py/test_smoke.py:19`](../py/test_smoke.py#L19) -> `dns_stats`; [`py/test_derived_fields.py:117`](../py/test_derived_fields.py#L117) -> `derived_fields`; [`scripts/inspect_mps_reference_contract.jl:1`](../scripts/inspect_mps_reference_contract.jl#L1); [`scripts/capture_mps_reference_environment.sh:1`](../scripts/capture_mps_reference_environment.sh#L1).
- **Language/environment:** Python/NumPy for tests; Julia standard library for text inspection; capture helper invokes Julia/Pkg/LinearAlgebra only if the configured Julia exists.
- **Inputs:** generated temporary tiny arrays for Python tests; source text for inspection; current process/module/Slurm state for capture.
- **Outputs/metrics:** assertion/console results, numbered source excerpts, and text environment inventory. No persistent output is hard-coded.
- **Slurm:** none.
- **Status:** **Structurally complete; not executed for this catalog.** These are the safest validation building blocks, but they do not establish a real-data reference or numerical tolerances. The baseline plan explicitly leaves test values and tolerances unresolved ([`validation/mps_v1_baseline_plan.md:10`](../validation/mps_v1_baseline_plan.md#L10)).
- **First clean workflow:** **Yes as validation scaffolding, not as the scientific workflow itself.**

## Helper-module coverage

The following files are not omitted workflows; they are locally imported libraries or data/config assets used by entries above:

- Modern shared stack: [`py/dns_stats.py`](../py/dns_stats.py), [`py/derived_fields.py`](../py/derived_fields.py), [`py/les_io.py`](../py/les_io.py), [`py/mps_io.py`](../py/mps_io.py), [`py/peps_io.py`](../py/peps_io.py), and [`py/save_tikz.py`](../py/save_tikz.py) are covered by B3-B4 and C1-C8.
- Legacy tensor/physics stack: A8-A10 cover the imported files under `mps_py_codes/`, including [`CritChis_uneq.py`](../mps_py_codes/CritChis_uneq.py), [`FindSingValsAndEntEntr.py`](../mps_py_codes/FindSingValsAndEntEntr.py), [`MPS_permute.py`](../mps_py_codes/MPS_permute.py), [`calculate_eps_stilde.py`](../mps_py_codes/calculate_eps_stilde.py), [`calculate_finite_difference.py`](../mps_py_codes/calculate_finite_difference.py), [`calculate_gradients.py`](../mps_py_codes/calculate_gradients.py), [`calculate_orientation.py`](../mps_py_codes/calculate_orientation.py), [`calculate_spectral_derivative.py`](../mps_py_codes/calculate_spectral_derivative.py), [`calculate_spectrum.py`](../mps_py_codes/calculate_spectrum.py), [`calculate_tkEnergy.py`](../mps_py_codes/calculate_tkEnergy.py), [`calculate_u_comp_fidelity_l2norm.py`](../mps_py_codes/calculate_u_comp_fidelity_l2norm.py), [`comp_ratio_uneq.py`](../mps_py_codes/comp_ratio_uneq.py), [`plotting.py`](../mps_py_codes/plotting.py), [`prime_sort.py`](../mps_py_codes/prime_sort.py), [`reshape_and_transpose.py`](../mps_py_codes/reshape_and_transpose.py), [`save_as_mat.py`](../mps_py_codes/save_as_mat.py), and [`velocity_name.py`](../mps_py_codes/velocity_name.py). [`combine_parts.py`](../mps_py_codes/combine_parts.py), [`compare_org_comp.py`](../mps_py_codes/compare_org_comp.py), and [`compute_URDNS.py`](../mps_py_codes/compute_URDNS.py) expose functions but have no CLI/main guard; they are potential future subcommands, not current standalone workflows.
- `mps_py_codes/config_FI.json` is a path registry, not a runnable workflow.
- `LES_interface/index.html` and `LES_interface/crc/index.html` are front ends for B1, not compute workflows.
- `LES_interface/crc/les_active_jobs.json`, `les_history.json`, and the empty legacy `LES_interface/les_history.json` are state/evidence files, not entry points.
- `codex_transcripts/` is ignored by Git and contains planning transcripts, not executable repository evidence.

## Best three first candidates

This ranking applies the requested five criteria without selecting a scientific convention or authorizing a run.

| Rank | Candidate | Reuse | Complete/evidenced path | Cost and safety | Rebuild risk | Value to later work | Why it ranks here |
|---:|---|---|---|---|---|---|---|
| 1 | **C1 DNS flame statistics + validation + plots** | Very high: baseline for DNS, MPS, LES, PEPS, filtered comparisons | End-to-end driver, core library, plotting, wrappers, README, and synthetic tests are present; real execution is unverified | Cube-only can be bounded; checked-in default `both` + validation is 180 GiB/2 h and should not be inherited blindly | High scientific value means silent formula/crop/operator changes would be costly | Establishes the result schema and DNS reference used throughout C2-C7 | Best first *scientific* standardization target once the user selects one dataset/timestep/region and accepts the existing formulas/operator. Start with one bounded case, not the default full campaign. |
| 2 | **B1 CRC block filter** | High across velocity, scalar, LES matching, and CR studies | Strongest run evidence: 29 completed CRC history records, compute/report path, Slurm template, and UI | Historical cases completed with small thread counts; resource fields are explicit. Repository packaging is incomplete but the repair is routine | Medium: choosing this block-average operator over the older same-grid filter is scientifically meaningful | Feeds filtered-DNS, energy, JHS spectrum, and LES-matching work | Best first *CRC mechanics* target after confirming B1's operator is intended. Preserve the recorded mean/variance/SGS residuals as invariants. |
| 3 | **C4 cross-source/cross-order comparison plotting** | High whenever any upstream source changes | Complete consumer/plotter chain and dependent Slurm wrapper; input schema is explicit | Low compute and no tensor rebuild; safest way to validate packaging and schema | Low implementation risk, though upstream semantic compatibility must be recorded | Immediately supports MPS/LES/PEPS/statistics iteration and exposes schema drift | Best low-cost integration target after one C1 reference pickle exists. It should not be used to declare scientific equivalence across differently prepared sources. |

### Why MPS compression is not ranked in the first three

A1/A2 are likely to be reused often and are central to the project, but repository evidence is insufficient to select an authoritative implementation or approve a rebuild: active scripts differ; ordering/bit significance, crop/layout, normalization, cutoff/chi semantics, environment, output contract, and tolerance remain unresolved. The provisional reference and validation plan already record these blockers ([`docs/mps_v1_decisions.md:10`](mps_v1_decisions.md#L10), [`validation/mps_v1_baseline_plan.md:1`](../validation/mps_v1_baseline_plan.md#L1)). Ranking a safe DNS/filter/reporting foundation ahead of MPS compression avoids silently deciding those scientific questions.

## Unresolved decisions intentionally not made

No question was required to finish this catalog because ambiguity can be recorded without choosing an authority. Before implementation or real-data execution, the relevant owner decisions still include:

- which of `0mps_calc.jl`, `mps_calc.jl`, or earlier Python/Julia variants is authoritative for each dataset;
- tensor site ordering and bit significance, crop/layout/endian/axis conventions, normalization, truncation and fidelity definitions, and regression observables;
- whether B1 block coarsening, B2 same-grid moving average, or another filter is the intended LES reference;
- DNS/LES timestep, region, field set, chi formula/mode, FD order, periodicity, units, PEPS orientation/sign/rescaling, and output tolerances;
- any CRC resource envelope or campaign size.

Those decisions have different scientific consequences and must not be inferred from active arrays, defaults, comments, or a visual diagnostic.

## Change confirmation

Single new file created: `docs/workflow_catalog.md`.

No historical code, notebook, Slurm file, environment, dataset, result, or existing documentation file was modified, moved, renamed, deleted, or executed. No package was installed and no job was submitted.
