# SLIC

Code and data for

> M. C. Chung, A. Zacharia and J. Guan, **A sample-size-invariant scoring criterion for sparse model discovery**, *Communications Physics* (under review).

Archived at Zenodo: [doi:10.5281/zenodo.23001594](https://doi.org/10.5281/zenodo.23001594). This DOI always resolves to the newest release.

SLIC (the sample-length-scaling information-like criterion) scores candidate models in sparse model discovery with a complexity penalty that grows in proportion to sample size, so that the balance between accuracy and sparsity does not shift as data accumulate.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | SLIC implementation and supporting routines. `sparse_regress.jl` was revised to support the effective sample size; `neff.jl`, `waic_gmdl.jl`, `no_projection.jl` and `enumeration.jl` were added in revision |
| `scripts/` | Julia scripts for the original analyses, including `sloshtank.jl` (sloshing-tank experiment) |
| `scripts/rev/` | Julia scripts for the analyses added in revision; `rev_common.jl` holds the shared setup |
| `cluster/` | Slurm job scripts that ran them (set `REPO_DIR` or submit from the repository root) |
| `scripts/figures_python/` | Python scripts that draw the revised figures, with `requirements.txt` |
| `data/exp_raw/` | Experimental sloshing-tank data (Bauerlein and Avila, *J. Fluid Mech.* 2021) |
| `data/results/` | Analysis results behind each figure, one folder per figure (`.jld`, readable with HDF5) |
| `data/source_data/` | Source data for the main-text figures (Supplementary Data 1-4, Excel) |

The results behind the figures are in `data/results/`. The folders `data/sims/ode_results_main/` and `data/sims/ode_results_si/` hold results from the first version of the manuscript and are kept for reference. Files for other analyses from earlier stages of the project (partial differential equations, RNA-nanoparticle and kernel analyses) are retained but are not used in this paper.

## Results and scripts for each figure

| Figure | Computed by (`scripts/rev/`) | Results in `data/results/` | Drawn by (`scripts/figures_python/`) |
| --- | --- | --- | --- |
| Fig. 1b, Supp. Fig. S1 | `fig1b_pool.py` (Python) | `fig1b_figS1/` (computed by `fig1b_pool.py`) | `gen_fig1b_panel.py`, `gen_s1.py` |
| Fig. 2, Supp. Fig. S2 | `rerun_fig2_neffall_r50.jl` | `fig2_benchmarks/` | `gen_fig2_grid.py` (Fig. 2 metrics grid), `gen_s2.py` (Supp. Fig. S2) |
| Fig. 3, Supp. Figs. S5-S8 | `rerun_si_neffall.jl` | `fig3_perturbations/` | `gen_fig3_panels.py` (Fig. 3 panels; run with `dts`, `Ts`, `ss` or `ps`), `gen_s5_s8.py` |
| Fig. 4, Supp. Fig. S13 | `scripts/sloshtank.jl` | `data/exp_raw/` | `scripts/sloshtank.jl`; `fig4_port.py` reproduces the Fig. 4 predictions in Python |
| Supp. Figs. S3-S4 | `rerun_waic_gmdl_neffall.jl` | `figS3_S4_waic_gmdl/` | `gen_s3_s4.py` |
| Supp. Fig. S9 | `rerun_noproj_scoreall.jl` | `figS9_no_projection/` | `gen_s9.py` |
| Supp. Fig. S10 | `rerun_longtraj_allsys.jl` | `figS10_long_trajectory/` | `gen_s10.py` |
| Supp. Fig. S11 | `rerun_duffing_fixed.jl` | `figS11_duffing/` | `gen_si_duffing.py` |
| Supp. Fig. S12 | `enumerate_all.jl`, after `calibrate_c.jl` (see `cluster/submit_enumerate_small.sh`) | `figS12_enumeration/` | `gen_s12.py` |
| Supp. Table S1 | `probe_neff_definitions.jl` | `tableS1_neff_definitions/` (job logs; cases 1-4) | |

The Julia scripts write their results to `data/sims/ode_results_rev/`; the copies used for the figures are in `data/results/`. The Python figure scripts read them directly and write to `plots/`; for example, from the repository root:

```
pip install -r scripts/figures_python/requirements.txt
python scripts/figures_python/fig1b_pool.py        # recomputes the Fig. 1b / Supp. Fig. S1 data (about 15 s)
python scripts/figures_python/gen_fig1b_panel.py
```

`mlib.py` holds the metric functions shared by several scripts, `paperstyle.py` the shared plotting style, and `tickcheck.py` checks that every axis's ticks enclose its plotted data. `gen_s5_s8.py` takes several minutes the first time, while it computes and caches the metrics.

## Running the Julia code

This code base uses the [Julia Language](https://julialang.org/) (version 1.10) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/). To reproduce the project locally:

1. Download this code base.
2. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This installs the package versions recorded in `Project.toml` and `Manifest.toml`. Most scripts start with

```julia
using DrWatson
@quickactivate "SLIC"
```

which activates the project and handles local paths.
