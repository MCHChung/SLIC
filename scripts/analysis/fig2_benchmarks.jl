using DrWatson
@quickactivate "SLIC"
using JLD, Random, Statistics

include(scriptsdir("analysis", "common.jl"))

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

include(scriptsdir("analysis", "sparse_regress_probe.jl"))   # score_on_all override

# ============================================================================
# Fig. 2 main benchmark under n_eff + score-on-all ("neff_all").
#
#   - scoring is on ALL n (score_on_all=true), not a 20% holdout, with η
#     rescaled so the conditioning floor is preserved (see sparse_regress_probe.jl);
#   - n_eff enters both fit and penalty (Position 1);
#   - per-system conditioning weight c (C_VALS below).
#
# Rationale: scoring on a 20% holdout would penalise the competitors at
# 0.2·n_eff (a ~3.7x handicap), and scoring at raw n would use an inflated sample
# size. neff_all scores every criterion at the effective sample size on all the
# data. SLIC is invariant to all of this (n cancels in n·log(k·RSS/n)); the
# competitors are not.
#
# Only the neff_all arm is saved.
#
# RUNS = 50. System-level recovery is an all-or-nothing per-run metric, so its
# standard error near p = 0.5 is ~0.07 at 50 runs (~0.10 at 25); 50 runs keeps
# chance non-monotone bumps in the recovery curves small.
#
# 36-task array: 6 systems × 6 noise levels.  task = sys_idx*6 + noise_idx
# ============================================================================

const NOISE_LEVELS = [0, 5, 10, 20, 30, 40]
const RUNS         = 50
const NUM_BATCHES  = 20
const TOL          = 0.7
const ICS          = ["slic", "aic", "aicc", "bic", "hqic", "bc", "kic"]
# Per-system c, matching discover_model_ode_main.jl.
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

    outdir = datadir("sims", "ode_results", "fig2_neffall_r50")
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
    error("Usage: julia fig2_benchmarks.jl SYS NOISE_IDX  (or SLURM_ARRAY_TASK_ID ∈ [0,35])")
end
main_for_sys_noise(sys, noise_idx)
