using DrWatson
@quickactivate "SLIC"
using JLD, Random, Statistics, LinearAlgebra

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
include(srcdir("generate_ode_data.jl"))

# ============================================================================
# DUFFING WEAK-CUBIC SWEEP WITH A FIXED INITIAL CONDITION
#
# WHY THE INITIAL CONDITION IS FIXED. The cubic term's share of the restoring
# force scales as alpha * A^2 with A the trajectory amplitude. If the initial
# condition were redrawn for every alpha, the amplitude would vary between runs
# and confound the very quantity being swept: the measured cubic contribution
# need not be monotonic in alpha, and individual alphas could draw unusually
# clean or unusually hard trajectories, neither of which is a property of alpha.
#
# u0 is therefore fixed at U0_FIXED for every alpha, and every alpha uses the
# same noise seeds, so alpha is the only quantity that varies across the sweep.
# The measured cubic ratio is then monotonic by construction and the panels can
# be plotted against it.
#
# The Duffing system is
#       xddot = -delta*xdot - omega^2*x - alpha*x^3
# with delta = 0.1 and omega = 1 held fixed (system 7 in generate_ode_data.jl,
# ps = [delta, omega, alpha]).
#
# One task per alpha (SLURM_ARRAY_TASK_ID 0-8, or the alpha index 1-9 as argument):
#   alpha = 0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0
# ============================================================================

const SYS_D     = 7
const DELTA     = 0.1
const OMEGA     = 1.0
const U0_FIXED  = [1.5, 0.0]        # identical for every alpha: the whole point
const ALPHAS    = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0]
const NOISE_D   = [0, 5, 10, 20]
const RUNS_D    = 25
const BATCH_D   = 20
const TOL_D     = 0.7
const C_D       = 1e-2
const ICS_D     = ["slic", "aic", "aicc", "bic", "hqic", "bc", "kic"]
const DT_D      = 1e-3
const T_D       = 60.0
const P_ORDER   = 10

"""Share of the restoring force carried by the cubic term, on the noise-free
trajectory. With u0 fixed this is monotonic in alpha."""
function cubic_share(X, alpha)
    x = X[1, :]
    f_lin  = (OMEGA^2) .* x
    f_cub  = alpha .* x .^ 3
    return var(f_cub) / var(f_lin .+ f_cub)
end

function run_alpha(alpha::Float64)
    tspan = (0.0, T_D)
    ts    = collect(0.0:DT_D:T_D)
    ps    = [DELTA, OMEGA, alpha]

    # single fixed trajectory, shared by every noise level and every run
    Xtrue, _ = GenData(SYS_D, tspan, DT_D, U0_FIXED, ps)
    ratio    = cubic_share(Xtrue, alpha)

    Lib = LIBS[SYS_D]
    # library width, built exactly as the run loop builds it
    ts0, x0, dx0 = DataWithFirstDeriv(smooth_ode(Xtrue[1, :])', ts, DT_D)
    n_lib = size(Lib(vcat(x0, dx0), ts0, 21; p=P_ORDER), 2)

    # true coefficient matrix: xdot = v ; vdot = -omega^2 x - delta v - alpha x^3
    Ξtrue = zeros(n_lib, 2)
    Ξtrue[2, 1] = 1.0                     # dx/dt = v
    Ξtrue[1, 2] = -OMEGA^2                # dv/dt = -omega^2 x
    Ξtrue[2, 2] = -DELTA                  #        - delta v
    Ξtrue[6, 2] = -alpha                  #        - alpha x^3   (library index 6 = x^3)

    println("=== alpha = $alpha   cubic share = $(round(100*ratio, digits=3))% ===")
    flush(stdout)

    results = Dict{Int, Any}()
    for NoisePct in NOISE_D
        Ξs = Dict(ic => zeros(n_lib, 2, RUNS_D) for ic in ICS_D)
        n_eff_run = zeros(RUNS_D)
        for run = 1:RUNS_D
            # identical seed sequence for every alpha, so the noise realizations
            # are matched across the sweep and only alpha differs
            Random.seed!(90210 + 1000*NoisePct + run)
            # Duffing is a second-order oscillator observed through x alone, so
            # the pipeline mirrors the Van der Pol branch: noise on x, smooth,
            # then recover the velocity by numerical differentiation.
            xtrue = Xtrue[1, :]
            η  = NoisePct*std(xtrue)/100
            xn = xtrue + η .* randn(length(xtrue))
            xsm = smooth_ode(xn)
            ts_tr, x, dx = DataWithFirstDeriv(xsm', ts, DT_D)
            Xsm = vcat(x, dx)
            _, _, wind = FindW(Xsm, ts_tr, ws=21:2:121, p=P_ORDER)
            qt = dwInt_sc(Xsm, ts_tr, wind, p=P_ORDER)
            θ  = Lib(Xsm, ts_tr, wind, p=P_ORDER)
            n_eff_val = min(size(θ,1),
                            n_eff_gaussian(ts_tr[end]-ts_tr[1], wind*DT_D; p=P_ORDER))
            n_eff_run[run] = n_eff_val
            for ic in ICS_D
                Ξ, _, _ = EnAdSR(θ, qt', ic; tol=TOL_D, c=C_D, num_batches=BATCH_D,
                                 n_eff=n_eff_val, fit_uses_neff=true, score_on_all=true)
                Ξs[ic][:, :, run] = Ξ
            end
        end
        results[NoisePct] = Dict("Xis" => Ξs, "n_eff" => n_eff_run)
        println("  noise $NoisePct% done"); flush(stdout)
    end

    outdir = datadir("sims", "ode_results", "duffing_fixed_u0")
    mkpath(outdir)
    outfile = joinpath(outdir, "duffing_alpha$(alpha)_fixedu0.jld")
    wsave(outfile, Dict{String,Any}(
        "alpha" => alpha, "delta" => DELTA, "omega" => OMEGA,
        "u0" => U0_FIXED, "cubic_ratio" => ratio,
        "noise_levels" => NOISE_D, "runs" => RUNS_D, "c" => C_D,
        "Xitrue" => Ξtrue, "results" => results))
    println("Wrote $outfile"); flush(stdout)
end

idx = haskey(ENV, "SLURM_ARRAY_TASK_ID") ? parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) + 1 :
      (length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1)
run_alpha(ALPHAS[idx])
