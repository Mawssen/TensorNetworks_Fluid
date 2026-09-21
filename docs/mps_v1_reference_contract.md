# MPS v1 provisional reference contract

## Status and scope

This document records, but does not endorse or correct, the behavior of `Julia/0mps_calc.jl` as the provisional numerical reference and `Julia/00mps_calc.slurm` as the provisional CRC wrapper. Their status as provisional references is a project decision supplied for this documentation task, not a conclusion drawn from the files.

The numerical contract below is limited to the active command-line path at the end of `0mps_calc.jl`; earlier examples and analysis blocks are commented out, while the active path begins at `total_time = @elapsed begin` and ends with an elapsed-time print ([`Julia/0mps_calc.jl:606-804`](../Julia/0mps_calc.jl#L606-L804), [`Julia/0mps_calc.jl:806-870`](../Julia/0mps_calc.jl#L806-L870)).

## Runtime and dependencies

- The numerical reference imports `ITensors`, `ITensorMPS`, `LinearAlgebra`, `MAT`, `HDF5`, `Random`, `JSON`, and `Printf`, and separately imports `MKL` ([`Julia/0mps_calc.jl:1-11`](../Julia/0mps_calc.jl#L1-L11)).
- It sets the BLAS thread count to 10 and the ITensors warning order to 40 at file scope ([`Julia/0mps_calc.jl:12-13`](../Julia/0mps_calc.jl#L12-L13)).
- The provisional wrapper exports `OMP_NUM_THREADS=10` and invokes a Julia executable at `$HOME/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia` ([`Julia/00mps_calc.slurm:18-19`](../Julia/00mps_calc.slurm#L18-L19)).
- **UNRESOLVED:** The exact Julia package versions, manifest, BLAS implementation/configuration observed at runtime, Julia thread count, loaded CRC modules, and Slurm environment are not recorded in either provisional source. Capture them with `scripts/capture_mps_reference_environment.sh` before producing a baseline; the sources only name imports and a Julia executable ([`Julia/0mps_calc.jl:1-13`](../Julia/0mps_calc.jl#L1-L13), [`Julia/00mps_calc.slurm:18-19`](../Julia/00mps_calc.slurm#L18-L19)).

## Command-line interface

- The active numerical entry point requires three positional arguments: input path, a mode string, and an integer split code ([`Julia/0mps_calc.jl:806-809`](../Julia/0mps_calc.jl#L806-L809)).
- A mode beginning case-insensitively with `chi=` is parsed as an integer maximum bond dimension; every other mode string is parsed as `Float64` cutoff ([`Julia/0mps_calc.jl:811-819`](../Julia/0mps_calc.jl#L811-L819)).
- The input basename must match `_(\d+)\.dat$`; the captured digits become the timestep and the output directory name `truncated_<timestep>` ([`Julia/0mps_calc.jl:821-824`](../Julia/0mps_calc.jl#L821-L824)).
- The wrapper calls the numerical reference as `julia 0mps_calc.jl <input-path> <mode> <split>` ([`Julia/00mps_calc.slurm:50-51`](../Julia/00mps_calc.slurm#L50-L51)).
- **UNRESOLVED:** The clean workflow's final command name, argument validation, path handling, and overwrite policy require an implementation decision; the historical entry point does not define those new-interface policies ([`Julia/0mps_calc.jl:806-824`](../Julia/0mps_calc.jl#L806-L824)).

## Active input interpretation and crop

- The active path hard-codes a full grid of `nx=864`, `ny=1008`, and `nz=576`, and a cubic retained extent `ntn=512` ([`Julia/0mps_calc.jl:825-829`](../Julia/0mps_calc.jl#L825-L829)).
- It reads exactly `4 * nx * ny * nz` bytes, reinterprets them first as `UInt32`, byte-swaps each word with `ntoh`, then reinterprets the result as `Float32` ([`Julia/0mps_calc.jl:831-835`](../Julia/0mps_calc.jl#L831-L835)).
- It reshapes the values to `(nx, ny, nz)` and takes 512 entries centered independently in each dimension, using one-based starts `div(size,2)-div(512,2)+1` and inclusive ranges ending at `start+511` ([`Julia/0mps_calc.jl:836-842`](../Julia/0mps_calc.jl#L836-L842)).
- **UNRESOLVED:** Whether the hard-coded dimensions, big-endian word interpretation, center crop, `Float32` input interpretation, and the names/order `x,y,z` are required scientific conventions or dataset-specific historical behavior must be decided before implementation ([`Julia/0mps_calc.jl:825-842`](../Julia/0mps_calc.jl#L825-L842)).
- **UNRESOLVED:** No units or physical quantity formula is attached to the active binary values; the wrapper supplies field-like filename tokens but no units or transformation formula ([`Julia/00mps_calc.slurm:21-26`](../Julia/00mps_calc.slurm#L21-L26), [`Julia/0mps_calc.jl:831-842`](../Julia/0mps_calc.jl#L831-L842)).

## Tensorization and ordering

- `ordering` computes `N` as `Int(log2(size(u,1)))`, takes `ndim` from `ndims(u)`, and constructs a target shape containing `ndim*N` dimensions of size 2 ([`Julia/0mps_calc.jl:317-320`](../Julia/0mps_calc.jl#L317-L320)).
- Split code 1 reshapes the input to that binary shape and permutes it with `perm_tuple(N, ndim)` ([`Julia/0mps_calc.jl:322-337`](../Julia/0mps_calc.jl#L322-L337)).
- `perm_tuple` builds an interleaving by taking position `i` from each contiguous dimension block before advancing to position `i+1`; `_block_starts` places those blocks at offsets `(k-1)*N+1` ([`Julia/0mps_calc.jl:181-194`](../Julia/0mps_calc.jl#L181-L194), [`Julia/0mps_calc.jl:207-211`](../Julia/0mps_calc.jl#L207-L211)).
- The inverse for split code 1 uses `inv_perm_tuple`, whose literal index sequence is `d*(i-1)+b` while iterating block `b` before position `i` ([`Julia/0mps_calc.jl:196-217`](../Julia/0mps_calc.jl#L196-L217), [`Julia/0mps_calc.jl:342-361`](../Julia/0mps_calc.jl#L342-L361)).
- The reference also defines split codes 2 through 6, including several distinct contiguous or middle-oriented permutations, although only split 1 is in the recommended v1 target ([`Julia/0mps_calc.jl:219-315`](../Julia/0mps_calc.jl#L219-L315), [`Julia/0mps_calc.jl:322-358`](../Julia/0mps_calc.jl#L322-L358)).
- **UNRESOLVED:** The semantic bit significance of each reshaped binary index, and whether the comments' `x1`, `x2`, and related labels denote most- or least-significant spatial bits, is not explicitly defined by the source ([`Julia/0mps_calc.jl:317-337`](../Julia/0mps_calc.jl#L317-L337), [`Julia/0mps_calc.jl:531-539`](../Julia/0mps_calc.jl#L531-L539)).
- **UNRESOLVED:** The active code does not assert that every input dimension is a power of two before reshaping; it derives `N` only from the first dimension ([`Julia/0mps_calc.jl:317-325`](../Julia/0mps_calc.jl#L317-L325)). Decide whether the clean workflow rejects nonconforming shapes or reproduces downstream reshape failure.

## MPS construction and truncation

- `mps_truncate` creates `ndim*N` `Qubit` site indices, flattens the ordered tensor with `reshape(u_perm, :)`, and constructs an MPS from that vector and those sites ([`Julia/0mps_calc.jl:366-379`](../Julia/0mps_calc.jl#L366-L379)).
- With neither truncation parameter, it calls `MPS(u_vector, s, cutoff=0)`; with cutoff only, `MPS(u_vector, s, cutoff=cutoff)`; with maximum dimension only, `MPS(u_vector, s, maxdim=χ, cutoff=0)`; and with both, `MPS(u_vector, s, cutoff=cutoff, maxdim=χ)` ([`Julia/0mps_calc.jl:381-394`](../Julia/0mps_calc.jl#L381-L394)).
- The active `chi=` path deliberately supplies both `χ=chi_val` and `cutoff=0.0`, while the cutoff path supplies only `cutoff=cf_val`; both pass the parsed split code ([`Julia/0mps_calc.jl:846-850`](../Julia/0mps_calc.jl#L846-L850)).
- The recorded `chi_crit` value is `linkdims(ψ)`, i.e. a sequence of link dimensions rather than a single scalar ([`Julia/0mps_calc.jl:398`](../Julia/0mps_calc.jl#L398)).
- Reconstruction contracts the MPS, converts it to an array over the site indices, applies the inverse ordering, and returns the reconstructed tensor with `chi_crit` when mutual information is not requested ([`Julia/0mps_calc.jl:400-412`](../Julia/0mps_calc.jl#L400-L412)).
- The active path does not explicitly normalize the input before MPS construction; normalization appears only inside the separate mutual-information helper ([`Julia/0mps_calc.jl:78-83`](../Julia/0mps_calc.jl#L78-L83), [`Julia/0mps_calc.jl:366-394`](../Julia/0mps_calc.jl#L366-L394)).
- **UNRESOLVED:** Required normalization, the scientific meaning of cutoff, truncation behavior across package versions, output dtype, and any fidelity/error definition must be approved before the clean workflow is implemented ([`Julia/0mps_calc.jl:366-413`](../Julia/0mps_calc.jl#L366-L413), [`Julia/0mps_calc.jl:831-850`](../Julia/0mps_calc.jl#L831-L850)).

## Output

- `save_to_mat` writes two MAT variables named `u` and `chi_crit` ([`Julia/0mps_calc.jl:416-425`](../Julia/0mps_calc.jl#L416-L425)).
- The active path tags split codes 1 through 4 as `il`, `seq`, `comb1`, and `combn`, and errors for any other split code at the filename stage ([`Julia/0mps_calc.jl:852-856`](../Julia/0mps_calc.jl#L852-L856)).
- Output names are `truncated_<input-basename>_512_<tag>_cf<value>.mat` for cutoff mode or `truncated_<input-basename>_512_<tag>_chi<value>.mat` for maximum-dimension mode, inside `truncated_<timestep>` ([`Julia/0mps_calc.jl:821-824`](../Julia/0mps_calc.jl#L821-L824), [`Julia/0mps_calc.jl:858-866`](../Julia/0mps_calc.jl#L858-L866)).
- **UNRESOLVED:** Which outputs constitute the regression contract—reconstructed values, shape, dtype, link dimensions, filenames, metadata, error metrics, or some subset—must be selected before a baseline is approved ([`Julia/0mps_calc.jl:398-412`](../Julia/0mps_calc.jl#L398-L412), [`Julia/0mps_calc.jl:858-866`](../Julia/0mps_calc.jl#L858-L866)).

## Provisional CRC wrapper

- The wrapper requests one node, one task, 20 CPUs per task, the `smp` cluster and partition, 64 GiB, two hours, and array indices `0-27` with at most 10 concurrent tasks ([`Julia/00mps_calc.slurm:2-11`](../Julia/00mps_calc.slurm#L2-L11)).
- Its active arrays contain 14 filenames, two modes (`4.1e-4` and `chi=93`), and only split code 1; its index arithmetic maps each array index across split, mode, and filename ([`Julia/00mps_calc.slurm:21-40`](../Julia/00mps_calc.slurm#L21-L40)).
- It derives the timestep from the final underscore-delimited filename component and constructs inputs as `./jet_<timestep>/<filename>` ([`Julia/00mps_calc.slurm:42-48`](../Julia/00mps_calc.slurm#L42-L48)).
- **UNRESOLVED:** The header comment describes “mixfrac” and “4 total tasks,” but the active filename/mode arrays describe 14 fields and 28 combinations; do not interpret the comment or active arrays as the intended future campaign without user direction ([`Julia/00mps_calc.slurm:13-16`](../Julia/00mps_calc.slurm#L13-L16), [`Julia/00mps_calc.slurm:21-35`](../Julia/00mps_calc.slurm#L21-L35)).
- **UNRESOLVED:** Future CRC resource requests and any sweep design require explicit approval; the provisional wrapper records historical resource and array choices only ([`Julia/00mps_calc.slurm:2-11`](../Julia/00mps_calc.slurm#L2-L11)).
