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

include(scriptsdir("analysis", "sparse_regress_probe.jl"))  # score_on_all override
include(srcdir("no_projection.jl"))

# ============================================================================
# No-projection diagnostic across ALL systems.
#
# Replaces the weak-form Galerkin derivative with pointwise finite differences
# (CalcDeriv, 4th-order) so n_eff = n trivially (iid setting). Tests whether
# SLIC's advantage survives WITHOUT the projection that inflates sample size.
#
# System categories:
#   First-order, fully observed (single differentiation):
#     Lorenz(1), Rossler(2), LV(3), Brusselator(4)
#   Second-order (VdP(5), Pendulum(6)): observe the FULL phase-space state
#     (position + velocity) and differentiate once ('known_v'). This is the
#     direct analogue of the full-state observation the first-order systems
#     get — every system observes its full state and takes a single FD.
#
# Task encoding (SLURM_ARRAY_TASK_ID): one task per system = sys-1 (0..5).
# Each task sweeps all noise levels internally.
# ============================================================================

# Noise grid matches the main benchmark for cross-experiment consistency.
# (With single differentiation on full-state observation, the pipeline handles
# the higher noise levels.)
const NOISE_LEVELS_NP = [0, 5, 10, 20, 30, 40]
const RUNS_NP = 25
const NUM_BATCHES_NP = 20
const TOL_NP = 0.7
const C_VALS_NP = Dict(1=>1e-1, 2=>1e-1, 3=>1e-3, 4=>1e-2, 5=>1e0, 6=>1e-2)
const ICS_NP = ["slic", "aic", "aicc", "bic", "hqic", "bc", "kic"]
const SUBSAMPLE_NP = haskey(ENV, "SLIC_NP_SUBSAMPLE") ? parse(Int, ENV["SLIC_NP_SUBSAMPLE"]) : 1

# All systems use FULL-STATE observation + single differentiation, for a
# uniform, apples-to-apples no-projection test:
#   - First-order systems: observe (x1..xn), differentiate each once.
#   - Second-order systems (VdP, Pendulum): observe the full phase-space state
#     (position AND velocity), differentiate each once. This is the direct
#     analogue of full-state observation used for the first-order systems, NOT
#     a special assumption. (The alternative — observing position only and
#     double-differentiating — tests a different, partial-observation problem
#     that the first-order systems never face, so it is excluded here.)
const SECOND_ORDER = Set([5, 6])   # VdP, Pendulum — run known_v (single FD) only

function run_condition(sys, data, Lib, NoisePct, assume_velocity_known, rng)
    """One (system, noise, condition): returns Dict of Ξ arrays per criterion."""
    fname, sysname, n_state, n_lib, dt, Tdefault = SYSTEMS[sys]
    Ξs = Dict(ic => zeros(n_lib, n_state, RUNS_NP) for ic in ICS_NP)
    n_actual = zeros(RUNS_NP)

    for run = 1:RUNS_NP
        qt, θ = GetInputsNoProj(sys, data["Xtrues"], data["ts"], Lib, NoisePct;
                                subsample=SUBSAMPLE_NP,
                                assume_velocity_known=assume_velocity_known)
        n_actual[run] = size(qt, 2)
        c_sys = C_VALS_NP[sys]
        for ic in ICS_NP
            # iid setting: n_eff = n, so no n_eff argument. Score on all n
            # (score_on_all=true) rather than the 20% holdout.
            Ξ, _, _ = EnAdSR(θ, qt', ic; tol=TOL_NP, c=c_sys,
                             num_batches=NUM_BATCHES_NP, score_on_all=true)
            Ξs[ic][:, :, run] = Ξ
        end
    end
    return Ξs, n_actual
end

function main(sys::Int)
    @assert 1 <= sys <= 6
    fname, sysname, n_state, n_lib, dt, Tdefault = SYSTEMS[sys]
    Lib = LIBS[sys]
    data = load(datadir("sims", "ode_data", fname))

    rng = Random.default_rng()
    Random.seed!(rng, 3000 + sys)

    println("=== No-projection (CalcDeriv): $sysname (sys=$sys) ===")
    flush(stdout)

    # Determine conditions. All systems: full-state observation, single
    # differentiation. For second-order systems that means known_v.
    conditions = sys in SECOND_ORDER ? ["known_v"] : ["fully_observed"]

    results = Dict{Int, Any}()
    for NoisePct in NOISE_LEVELS_NP
        println("  noise = $NoisePct%"); flush(stdout)
        per_cond = Dict{String, Any}()
        for cond in conditions
            avk = (cond == "known_v")
            Ξs, n_actual = run_condition(sys, data, Lib, NoisePct, avk, rng)
            per_cond[cond] = Dict("Xis" => Ξs, "n" => n_actual)
        end
        results[NoisePct] = per_cond
    end

    outdir = datadir("sims", "ode_results", "no_projection_scoreall_sub$(SUBSAMPLE_NP)")
    mkpath(outdir)
    outfile = joinpath(outdir, "$(lowercase(replace(sysname, " " => "_")))_noproj_results.jld")

    wsave(outfile, Dict{String, Any}(
        "sysname"      => sysname,
        "sys"          => sys,
        "Xitrue"       => data["Ξtrue"],
        "noise_levels" => NOISE_LEVELS_NP,
        "conditions"   => conditions,
        "subsample"    => SUBSAMPLE_NP,
        "results"      => results,
    ))
    println("Wrote $outfile"); flush(stdout)
end

sys = if haskey(ENV, "SLURM_ARRAY_TASK_ID")
    parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) + 1
elseif length(ARGS) >= 1
    parse(Int, ARGS[1])
else
    error("Usage: julia no_projection_benchmarks.jl SYS  (or SLURM_ARRAY_TASK_ID ∈ [0,5])")
end

main(sys)
