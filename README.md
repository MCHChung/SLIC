# SLIC

Code and data for

> M. C. Chung, A. Zacharia and J. Guan, **A sample-size-invariant scoring criterion for sparse model discovery**, *Communications Physics* (under review).

Archived at Zenodo: [doi:10.5281/zenodo.23001594](https://doi.org/10.5281/zenodo.23001594). This DOI always resolves to the newest release.

SLIC (the sample-length-scaling information-like criterion) scores candidate models in sparse model discovery with a complexity penalty that grows in proportion to sample size, so that the balance between accuracy and sparsity does not shift as data accumulate.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | SLIC implementation and supporting routines, including the effective sample size (`neff.jl`), WAIC and gMDL (`waic_gmdl.jl`), the no-projection pipeline (`no_projection.jl`) and exhaustive enumeration (`enumeration.jl`) |
| `scripts/` | Julia scripts for data generation, model discovery and the sloshing-tank experiment (`sloshtank.jl`) |
| `scripts/analysis/` | Julia scripts that compute the results in `data/results/`; `common.jl` holds the shared setup |
| `scripts/figures_python/` | Python scripts that draw the figures, with `requirements.txt` |
| `data/exp_raw/` | Experimental sloshing-tank data (Bauerlein and Avila, *J. Fluid Mech.* 2021) |
| `data/results/` | Analysis results behind each figure, one folder per figure (`.jld`, readable with HDF5) |
| `data/source_data/` | Source data for the main-text figures (Supplementary Data 1-4, Excel) |

The results behind the figures are in `data/results/`. The folders `data/sims/ode_results_main/` and `data/sims/ode_results_si/` hold results from earlier runs of the benchmarks and are kept for reference. Files for other analyses from earlier stages of the project (partial differential equations, RNA-nanoparticle and kernel analyses) are retained but are not used in this paper.

## Results and scripts for each figure

| Figure | Computed by (`scripts/analysis/`) | Results in `data/results/` | Drawn by (`scripts/figures_python/`) |
| --- | --- | --- | --- |
| Fig. 1b, Supp. Fig. S1 | `fig1b_pool.py` (Python) | `fig1b_figS1/` (computed by `fig1b_pool.py`) | `gen_fig1b_panel.py`, `gen_s1.py` |
| Fig. 2, Supp. Fig. S2 | `fig2_benchmarks.jl` | `fig2_benchmarks/` | `gen_fig2_grid.py` (Fig. 2 metrics grid), `gen_s2.py` (Supp. Fig. S2) |
| Fig. 3, Supp. Figs. S5-S8 | `fig3_perturbations.jl` | `fig3_perturbations/` | `gen_fig3_panels.py` (Fig. 3 panels; run with `dts`, `Ts`, `ss` or `ps`), `gen_s5_s8.py` |
| Fig. 4, Supp. Fig. S13 | `scripts/sloshtank.jl` | `data/exp_raw/` | `scripts/sloshtank.jl`; `fig4_port.py` reproduces the Fig. 4 predictions in Python |
| Supp. Figs. S3-S4 | `waic_gmdl_benchmarks.jl` | `figS3_S4_waic_gmdl/` | `gen_s3_s4.py` |
| Supp. Fig. S9 | `no_projection_benchmarks.jl` | `figS9_no_projection/` | `gen_s9.py` |
| Supp. Fig. S10 | `long_trajectory.jl` | `figS10_long_trajectory/` | `gen_s10.py` |
| Supp. Fig. S11 | `duffing_sweep.jl` | `figS11_duffing/` | `gen_si_duffing.py` |
| Supp. Fig. S12 | `enumerate_all.jl`, after `calibrate_c.jl` (see [Running the analysis scripts](#running-the-analysis-scripts)) | `figS12_enumeration/` | `gen_s12.py` |
| Supp. Table S1 | `probe_neff_definitions.jl` | `tableS1_neff_definitions/` (job logs; cases 1-4) | |

The Julia scripts write their results to `data/sims/ode_results/`; the copies used for the figures are in `data/results/`. The Python figure scripts read them directly and write to `plots/`; for example, from the repository root:

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

## Running the analysis scripts

Each script in `scripts/analysis/` computes one task of a sweep. The task is chosen by the environment variable `SLURM_ARRAY_TASK_ID`, so a whole sweep can be submitted as a Slurm array job (for example `sbatch --array=0-35`), and a single task can be run locally from the repository root:

```
SLURM_ARRAY_TASK_ID=0 julia --project=. scripts/analysis/fig2_benchmarks.jl
```

Systems are numbered 1-6 (Lorenz, Rössler, Lotka-Volterra, Brusselator, Van der Pol, nonlinear pendulum) and noise levels 1-6 (0, 5, 10, 20, 30, 40 %). Where a task combines the two, task = 6 × (system - 1) + (noise level - 1).

| Script | Tasks | Each task is | Other settings |
| --- | --- | --- | --- |
| `fig2_benchmarks.jl` | 0-35 | a system and noise level | |
| `fig3_perturbations.jl` | 0-35 (0-23 for `ps`) | a system and noise level | `SLIC_SI_COND`: `dts` (panel a), `Ts` (b), `ss` (c) or `ps` (d) |
| `waic_gmdl_benchmarks.jl` | 0-35 | a system and noise level | |
| `no_projection_benchmarks.jl` | 0-5 | a system (all noise levels) | |
| `long_trajectory.jl` | 0-35 | a system and trajectory-length factor L = 1, 2, 5, 10, 20, 50 (task = 6 × (system - 1) + L index) | |
| `duffing_sweep.jl` | 0-8 | a value of alpha (0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2) | |
| `probe_neff_definitions.jl` | 0-5 | a case (0-3 give Supp. Table S1; 4-5 are the dt/5 oversampling test) | |
| `calibrate_c.jl`, `calibrate_gamma.jl` | | | `SLIC_CAL_SYS` = 1-6, one run per system |
| `enumerate_all.jl` | 0-11 (0-5 for the pendulum) | a noise level and equation, task = (number of equations) × (noise level - 1) + (equation - 1) | `SLIC_ENUM_SYS` = 3-6 and `SLIC_GAMMA=100`; run `calibrate_c.jl` for that system first |

The larger sweeps were run on a cluster with 8 cores and 32-48 GB of memory per task (16 cores and 128 GB for `long_trajectory.jl`), with up to a day per task (three days for `long_trajectory.jl`).
