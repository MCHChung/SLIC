using DrWatson
@quickactivate "SLIC"
using JLD, Random, Statistics

include(scriptsdir("rev", "rev_common.jl"))

# ----------------------------------------------------------------------------
# Note on the residual floor eta (applies identically to every criterion,
# SLIC included). Candidates are scored on RSS + eta rather than RSS; see
# score() in src/sparse_regress.jl. In the main pipeline eta = c*cond(theta),
# with the per-system constant c passed to EnAdSR. In the exhaustive
# enumeration (enumerate_all.jl) eta = gamma*RSS0, where RSS0 is the true
# model's residual on noise-free data (calibrate_c.jl) and gamma = 100
# (SLIC_GAMMA). The floor is numerical: as the noise goes to zero, RSS goes to
# zero, which would otherwise make the log-likelihood terms diverge and leave
# the fit term free to fall with every added term, so that most criteria would
# not sparsify at low noise.
# ----------------------------------------------------------------------------

include(scriptsdir("rev", "sparse_regress_probe.jl"))   # score_on_all override; src/ untouched

# ============================================================================
# Fig. 2 main benchmark, re-run under n_eff + score-on-all ("neff_all").
#
# Change vs rerun_with_neff_v2.jl:
#   - scoring is on ALL n (score_on_all=true), not the 20% holdout, with η
#     rescaled so the conditioning floor is preserved (see sparse_regress_probe.jl);
#   - n_eff enters both fit and penalty (Position 1);
#   - per-system c (the old driver used a flat CC=1e-2, wrong for 4/6 systems).
#
# Rationale: the published Fig. 2 (raw n, 20% holdout) penalises the competitors
# at 0.2·n_eff — a ~3.7x handicap — AND at the wrong (raw, inflated) n. neff_all
# is the single-correction, honest comparison the reviewer asked for: it scores
# every criterion at the effective sample size on all the data. SLIC is invariant
# to all of this (n cancels in n·log(k·RSS/n)); the competitors are not.
#
# Only the neff_all arm is saved. The split arms already exist: Fig. 2's
# score_on_all sweep (ode_results_rev/score_on_all) holds raw/neff × split/all at
# matched per-system c, and neff_v2 holds the Position-1 split arm.
#
# RUNS = 50 (was 25). System-level recovery is an all-or-nothing per-run metric,
# so at 25 runs its standard error is ~0.10 near p = 0.5 and roughly one curve in
# ten shows a visible non-monotone bump purely by chance. A monotonicity test
# against an isotonic null confirms none of the observed bumps are significant
# (all p > 0.10), i.e. the 25-run results are correct, not biased; 50 runs simply
# removes about three quarters of the visual raggedness in the recovery panel
# that Reviewer 2 asked us to add.
#
# 36-task array: 6 systems × 6 noise levels.  task = sys_idx*6 + noise_idx
# ============================================================================

const NOISE_LEVELS = [0, 5, 10, 20, 30, 40]
const RUNS         = 50
const NUM_BATCHES  = 20
const TOL          = 0.7
const ICS          = ["slic", "aic", "aicc", "bic", "hqic", "bc", "kic"]
# Per-system c, matching discover_model_ode_main.jl / the published Fig. 2 files.
const C_VALS = Dict(1=>1e-1, 2=>1e-1, 3=>1e-3, 4=>1e-2, 5=>1e0, 6=>1e-2)

function main_for_sys_noise(sys::Int, noise_idx::Int)
    @assert 1 <= sys <= 6
    NoisePct = NOISE_LEVELS[noise_idx]
    fname, sysname, n_state, n_lib, dt, Tdefault = SYSTEMS[sys]
    Lib   = LIBS[sys]
    c_sys = C_VALS[sys]

    println("=== Fig2 neff_all: $sysname (sys=$sys), noise=$NoisePct%, c=$c_sys ===")
    flush(stdout)

    data = load(datadir("sims", "ode_data", fname))
    Ξs = Dict(ic => zeros(n_lib, n_state, RUNS) for ic in ICS)
    n_eff_per_run = zeros(RUNS)

    Random.seed!(1500 + sys*100 + noise_idx)

    for run = 1:RUNS
        println("  run $run/$RUNS"); flush(stdout)
        qt, θ, n_eff_val = GetInputsWithNeff(sys, data["Xtrues"], data["ts"], Lib, NoisePct)
        n_eff_per_run[run] = n_eff_val
        for ic in ICS
            Ξ, _, _ = EnAdSR(θ, qt', ic; tol=TOL, c=c_sys, num_batches=NUM_BATCHES,
                             n_eff=n_eff_val, fit_uses_neff=true, score_on_all=true)
            Ξs[ic][:, :, run] = Ξ
        end
    end

    outdir = datadir("sims", "ode_results_rev", "fig2_neffall_r50")
    mkpath(outdir)
    outfile = joinpath(outdir,
        "$(lowercase(replace(sysname, " " => "_")))_neffall_noise$(noise_idx)_results.jld")
    out = Dict{String,Any}(
        "sysname"=>sysname, "sys"=>sys, "NoisePct"=>NoisePct, "runs"=>RUNS,
        "c"=>c_sys, "n_eff_per_run"=>n_eff_per_run, "n_eff_mean"=>mean(n_eff_per_run),
        "Xitrue"=>data["Ξtrue"], "ts"=>collect(data["ts"]))
    for ic in ICS
        out["Xis_neffall_$ic"] = Ξs[ic]
    end
    wsave(outfile, out)
    println("Wrote $outfile"); flush(stdout)
end

sys, noise_idx = if haskey(ENV, "SLURM_ARRAY_TASK_ID")
    task = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    (task ÷ 6) + 1, (task % 6) + 1
elseif length(ARGS) >= 2
    parse(Int, ARGS[1]), parse(Int, ARGS[2])
else
    error("Usage: julia rerun_fig2_neffall.jl SYS NOISE_IDX  (or SLURM_ARRAY_TASK_ID ∈ [0,35])")
end
main_for_sys_noise(sys, noise_idx)
