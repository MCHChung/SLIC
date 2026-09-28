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

include(scriptsdir("rev", "sparse_regress_probe.jl"))  # score_on_all override
include(srcdir("generate_ode_data.jl"))

# ============================================================================
# Long-trajectory scaling experiment (R2 §3 headline), v2.
#
# Changes vs v1:
#   - Extended L grid: {1, 2, 5, 10, 20, 50} (was {1,2,5,10,20})
#   - Position-1 scoring (fit_uses_neff=true): competitors get n_eff in BOTH
#     the fit-term leading coefficient and the penalty (coherent correction).
#     SLIC unchanged (invariant).
#   - Design matches on L (physical trajectory-length multiplier), NOT on a
#     target n_eff. n_eff is reported as a measured outcome for the x-axis.
#
# Task encoding (SLURM_ARRAY_TASK_ID): (sys_index)*len(L) + L_index.
#   2 systems × 6 L values = 12 tasks (0..11).
# ============================================================================

const L_FACTORS = [1, 2, 5, 10, 20, 50]
const NOISE_LEVELS_LT = [5, 20]
const RUNS_LT = 25
# Long trajectories are the expensive cells (Lorenz L=50 timed out three times
# at the 4-day wall). Reduce the run count only there; L <= 10 keeps 25 runs so
# it stays directly comparable with every other experiment.
runs_for(L) = L >= 50 ? 5 : (L >= 20 ? 10 : 20)
const NUM_BATCHES_LT = 20
const TOL_LT = 0.7
const C_VALS_LT = Dict(1=>1e-1, 2=>1e-1, 3=>1e-3, 4=>1e-2, 5=>1e0, 6=>1e-2)
const ICS_LT = ["slic", "aic", "aicc", "bic", "hqic", "bc", "kic"]
const SYSTEMS_LT = [1, 2, 3, 4, 5, 6]   # all six benchmark systems
const FIT_USES_NEFF = true  # Position 1

function gen_data_factor(sys::Int, L::Real, Tdefault::Real, dt::Real; rng=Random.default_rng())
    Xs = []
    tspan = (0.0, L * Tdefault)
    ts = collect(0.0:dt:tspan[2])
    for i = 1:3
        u0 = _initial_condition(sys, rng)
        Xtrue, _ = GenData(sys, tspan, dt, u0)
        push!(Xs, Xtrue)
    end
    return Xs, ts
end

function _initial_condition(sys::Int, rng)
    # Initial conditions copied verbatim from scripts/ode_data.jl (the script
    # that generated the stored benchmark data), so long trajectories live on
    # exactly the same attractors as every other experiment in the paper.
    if sys == 1          # Lorenz
        return Float64.(rand(rng, -50:50, 3))
    elseif sys == 2      # Rossler
        return Float64.(rand(rng, 0:5, 3))
    elseif sys == 3      # Lotka-Volterra
        return Float64.(rand(rng, 1:50, 2))
    elseif sys == 4      # Brusselator: [0, u] with u ~ U(0,10)
        return [0.0, Float64(rand(rng, 0.0:1.0:10.0))]
    elseif sys == 5      # Van der Pol
        return Float64.(rand(rng, 0:0.5:3, 2))
    elseif sys == 6      # Nonlinear pendulum: [θ, 0] released from rest
        return [Float64(rand(rng, π/2:0.1:3π/4)), 0.0]
    else
        error("_initial_condition: unsupported sys=$sys")
    end
end

function main(sys::Int, L_idx::Int)
    @assert sys in SYSTEMS_LT
    @assert 1 <= L_idx <= length(L_FACTORS)
    L = L_FACTORS[L_idx]

    fname, sysname, n_state, n_lib, dt, Tdefault = SYSTEMS[sys]
    Lib = LIBS[sys]
    data_existing = load(datadir("sims", "ode_data", fname))
    Ξtrue = data_existing["Ξtrue"]

    rng = Random.default_rng()
    Random.seed!(rng, 2500 + sys * 100 + L_idx)

    println("=== long_trajectory v2: $sysname, L=$L (length $(L*Tdefault)s), Position 1 ===")
    flush(stdout)

    results_per_noise = Dict{Int, Any}()

    for NoisePct in NOISE_LEVELS_LT
        println("  noise = $NoisePct%"); flush(stdout)
        nruns = runs_for(L)
        Ξs_run = Dict(ic => zeros(n_lib, n_state, nruns) for ic in ICS_LT)
        n_eff_run = zeros(nruns)

        for run = 1:nruns
            Xs, ts = gen_data_factor(sys, L, Tdefault, dt; rng=rng)
            qt, θ, n_eff_val = GetInputsWithNeff(sys, Xs, ts, Lib, NoisePct)
            n_eff_run[run] = n_eff_val

            c_sys = C_VALS_LT[sys]
            for ic in ICS_LT
                Ξ, _, _ = EnAdSR(θ, qt', ic;
                                 tol=TOL_LT, c=c_sys, num_batches=NUM_BATCHES_LT,
                                 n_eff=n_eff_val, fit_uses_neff=true, score_on_all=true)
                Ξs_run[ic][:, :, run] = Ξ
            end
        end
        results_per_noise[NoisePct] = Dict("Xis" => Ξs_run, "n_eff" => n_eff_run)
    end

    outdir = datadir("sims", "ode_results_rev", "long_trajectory_neffall_all_v2")
    mkpath(outdir)
    outfile = joinpath(outdir, "$(lowercase(replace(sysname, " " => "_")))_L$(L)_results.jld")

    wsave(outfile, Dict{String, Any}(
        "sysname"        => sysname,
        "sys"            => sys,
        "L_factor"       => L,
        "noise_levels"   => NOISE_LEVELS_LT,
        "Xitrue"         => Ξtrue,
        "fit_uses_neff"  => FIT_USES_NEFF,
        "results"        => results_per_noise,
    ))
    println("Wrote $outfile"); flush(stdout)
end

sys, L_idx = if haskey(ENV, "SLURM_ARRAY_TASK_ID")
    task = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    s_idx = task ÷ length(L_FACTORS)
    li = (task % length(L_FACTORS)) + 1
    SYSTEMS_LT[s_idx + 1], li
elseif length(ARGS) >= 2
    parse(Int, ARGS[1]), parse(Int, ARGS[2])
else
    error("Usage: julia long_trajectory_v2.jl SYS L_IDX  (or SLURM_ARRAY_TASK_ID ∈ [0,11])")
end

main(sys, L_idx)
