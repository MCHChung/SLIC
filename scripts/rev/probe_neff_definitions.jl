using DrWatson
@quickactivate "SLIC"
using JLD, Random, Statistics, LinearAlgebra

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

include(scriptsdir("rev", "sparse_regress_probe.jl"))   # score_on_all override
include(srcdir("neff.jl"))                              # the three n_eff definitions

# ============================================================================
# ROBUSTNESS OF THE CONCLUSIONS TO THE DEFINITION OF n_eff   (Reviewer 2, §3)
#
# The response letter states that the recomputation was verified against the
# more rigorous autocorrelation / projection-trace definitions. This probe
# supplies that evidence.
#
# Three definitions are compared, all implemented in src/neff.jl:
#
#   (1) gaussian  n_eff = T_total * sqrt(p/pi) / T_w        [headline]
#       with T_w = wind*dt. Since T_total = n*dt, the dt cancels and
#       n_eff = 1.784 * n / wind, i.e. the deflation is set by the window
#       size IN SAMPLES (wind in 21:2:121  ->  n/n_eff in 11.8 .. 67.8).
#
#   (2) autocorr  n_eff = n_proj / tau_eff, tau_eff from the integrated
#       autocorrelation of the PROJECTED residuals (Sokal windowing).
#       Residuals are taken from an OLS fit on the true support, so they
#       estimate the projected noise process rather than model error.
#
#   (3) trace     n_eff = n_proj * (sum w^2)/(sum w)^2 from the kernel
#       weights, the projection-operator trace under row normalisation.
#
# WHY THIS MATTERS BEYOND THE LETTER. Definition (1) does not saturate under
# oversampling: FindW selects the window in SAMPLES, so as dt shrinks the
# window's physical width T_w = wind*dt shrinks with it and n_eff keeps
# growing (empirically n_eff ~ n^0.86 across the dts sweep) where a
# saturating count would flatten. Whether that is correct depends on which
# correlation time is the relevant one: for the Gaussian likelihood the
# projected NOISE decorrelates on T_w, which supports (1); if instead the
# DYNAMICS must decorrelate, the honest count would be smaller still.
# Definition (2) measures the actual residual correlation and therefore
# settles the question empirically.
#
# Only the fully observed systems (1 Lorenz, 2 Rossler, 3 Lotka-Volterra,
# 4 Brusselator) are used. Van der Pol and the pendulum are observed through a
# single state and require the differentiate-the-velocity branch of
# GetInputsWithNeff, which this probe does not reproduce; including them here
# would silently prepare their data the wrong way.
#
# The probe reports, for each cell and each definition: n_eff itself, and the
# resulting selection performance for SLIC and the strongest competitors.
# SLIC is invariant to n_eff by construction (n cancels in n*log(k*RSS/n)),
# so its row is a control: it must not move.
#
# COST. The projected design is built once per run and reused for all three
# definitions, so the sweep is 10 runs x 3 noise levels x 4 criteria x 3
# definitions = 360 fits and only 30 projections per case. The ensemble is 12
# batches rather than the 20 used for the headline figures; this is a
# consistency check between definitions, all of which see identical data, so a
# smaller ensemble does not affect the comparison.
#
#   sbatch --array=0-5 cluster/submit_probe_neff_definitions.sh
# ============================================================================

const RUNS_P  = 10
const BATCH_P = 12
const TOL_P   = 0.7
const ICS_P   = ["slic", "bic", "kic", "aic"]
const NOISE_P = [0, 20, 40]
const C_VALS  = Dict(1=>1e-1, 2=>1e-1, 3=>1e-3, 4=>1e-2, 5=>1e0, 6=>1e-2)

# One cell per system; the dts sweep is where definition (1) is most exposed,
# so the Lorenz/Rossler cases are run at the finest sampling as well.
const CASES = [
    (1, "baseline", "Lorenz, standard sampling"),
    (2, "baseline", "Rossler, standard sampling"),
    (3, "baseline", "Lotka-Volterra, standard sampling"),
    (4, "baseline", "Brusselator, standard sampling"),
    (1, "fine",     "Lorenz, finest sampling (dt/5): oversampling stress test"),
    (2, "fine",     "Rossler, finest sampling (dt/5): oversampling stress test"),
]

"""Projected inputs plus all three n_eff definitions for one trajectory set."""
function inputs_all_neff(sys::Int, Xs, ts, Lib::Function, NoisePct; p::Int=10, dt_div::Int=1)
    # optional resampling to a finer grid for the oversampling stress test
    if dt_div > 1
        m  = (length(ts) - 1)*dt_div + 1
        ts = collect(range(ts[1], ts[end], length=m))
        Xs_r = Any[]
        for X in Xs
            ns = size(X, 1)
            Xr = Matrix{Float64}(undef, ns, m)
            for i = 1:ns
                Xr[i, :] = _interp_row(vec(X[i, :]), m)
            end
            push!(Xs_r, Xr)
        end
        Xs = Xs_r
    end

    # Noise is added explicitly and then smoothed, matching GetInputsWithNeff:
    # smooth_ode takes the matrix alone, not a noise percentage.
    function _prep(X)
        η = NoisePct*mean(std(X, dims=2))/100
        return smooth_ode(X + η .* randn(size(X)))
    end

    Xsm = _prep(Xs[1])
    _, _, wind = FindW(Xsm, ts, ws=21:2:121, p=p)
    qt = dwInt_sc(Xsm, ts, wind, p=p)
    θ  = Lib(Xsm, ts, wind, p=p)
    for i = 2:length(Xs)
        Xsm_i = _prep(Xs[i])
        # dwInt_sc returns (n_state x n_proj) and Lib returns (n_proj x n_lib),
        # so trajectories concatenate along different axes: hcat for qt, vcat
        # for theta. This matches GetInputsWithNeff exactly.
        qt = hcat(qt, dwInt_sc(Xsm_i, ts, wind, p=p))
        θ  = vcat(θ,  Lib(Xsm_i, ts, wind, p=p))
    end

    dt      = ts[2] - ts[1]
    T_total = ts[end] - ts[1]
    n_proj  = size(θ, 1)

    # (1) headline
    ne_gauss = min(n_proj, n_eff_gaussian(T_total, wind*dt; p=p) * length(Xs))

    # (2) empirical, from residuals of an OLS fit on the full library
    #     (the true support is not known to the criterion; using the full-library
    #     residual keeps this a property of the noise, not of model choice)
    # qt may be stored with states along either axis; pick the orientation whose
    # row count matches the design. n_eff_autocorr expects (n_obs x n_state).
    Y = size(qt, 1) == n_proj ? Matrix(qt) : Matrix(qt')
    @assert size(Y, 1) == n_proj "qt orientation does not match theta: size(qt) = $(size(qt)), n_proj = $n_proj"
    # The autocorrelation estimate is the only fragile one of the three; if it
    # fails, fall back to NaN so the other two definitions are still reported
    # rather than losing the whole case.
    ne_auto = try
        Ξ_ols = θ \ Y
        resid = Y - θ * Ξ_ols
        min(n_proj, n_eff_autocorr(resid))
    catch err
        @warn "n_eff_autocorr failed" exception=err size_qt=size(qt) size_theta=size(θ)
        NaN
    end

    # (3) projection-operator trace
    w = polynomial_kernel_weights(wind; p=p)
    ne_trace = min(n_proj, n_eff_trace(w, n_proj))

    return qt, θ, (gaussian=ne_gauss, autocorr=ne_auto, trace=ne_trace), n_proj, wind
end

_interp_row(v, m) = [v[clamp(round(Int, 1 + (k-1)*(length(v)-1)/(m-1)), 1, length(v))] for k = 1:m]

function metrics(Ξs, Ξtrue)
    tnz = abs.(Ξtrue) .> 1e-12
    rec = Float64[]; fpr = Float64[]; nsel = Float64[]
    for r = 1:size(Ξs, 3)
        p  = abs.(Ξs[:, :, r]) .> 1e-12
        FP = count(p .& .!tnz); TN = count(.!p .& .!tnz)
        push!(rec, all(p .== tnz) ? 1.0 : 0.0)
        push!(fpr, (FP + TN) > 0 ? FP/(FP+TN) : 0.0)
        push!(nsel, count(p))
    end
    return mean(rec), mean(fpr), mean(nsel)
end

function main(case_idx::Int)
    sys, mode, note = CASES[case_idx]
    fname, sysname, n_state, n_lib, dt, Tdefault = SYSTEMS[sys]
    Lib   = LIBS[sys]
    c_sys = C_VALS[sys]
    dt_div = mode == "fine" ? 5 : 1
    data  = load(datadir("sims", "ode_data", fname))

    println("="^80)
    println("n_eff DEFINITION ROBUSTNESS  case $case_idx: $note")
    println("="^80)

    for NoisePct in NOISE_P
        println("\n--- noise = $NoisePct% ---")
        Ξtrue = data["Ξtrue"]
        # collect the three n_eff values once (they do not depend on the criterion)
        Random.seed!(4242)
        _, _, nes, n_proj, wind = inputs_all_neff(sys, data["Xtrues"], data["ts"], Lib, NoisePct;
                                                  dt_div=dt_div)
        println("  n_proj = $n_proj, window = $wind samples")
        println("  n_eff:  gaussian = $(round(nes.gaussian,digits=1))  " *
                "autocorr = $(round(nes.autocorr,digits=1))  " *
                "trace = $(round(nes.trace,digits=1))")
        println("  n_proj/n_eff:  gaussian = $(round(n_proj/nes.gaussian,digits=1))x  " *
                "autocorr = $(round(n_proj/nes.autocorr,digits=1))x  " *
                "trace = $(round(n_proj/nes.trace,digits=1))x")

        # The projected design does not depend on which n_eff definition is used,
        # only the scalar penalty argument does. Preparing the data once per run
        # and reusing it for all three definitions cuts the projection cost by a
        # factor of three and, as a side benefit, guarantees that the three
        # definitions are compared on identical data rather than on independent
        # noise draws that merely share a seed.
        DEFS = ["gaussian", "autocorr", "trace"]
        Ξs = Dict((d, ic) => zeros(n_lib, n_state, RUNS_P) for d in DEFS, ic in ICS_P)
        for run = 1:RUNS_P
            Random.seed!(4242 + run)
            qt, θ, ne, _, _ = inputs_all_neff(sys, data["Xtrues"], data["ts"], Lib, NoisePct;
                                              dt_div=dt_div)
            nevs = Dict("gaussian" => ne.gaussian, "autocorr" => ne.autocorr,
                        "trace" => ne.trace)
            for d in DEFS, ic in ICS_P
                isnan(nevs[d]) && continue      # definition unavailable for this cell
                Ξ, _, _ = EnAdSR(θ, qt', ic; tol=TOL_P, c=c_sys, num_batches=BATCH_P,
                                 n_eff=nevs[d], fit_uses_neff=true, score_on_all=true)
                Ξs[(d, ic)][:, :, run] = Ξ
            end
        end
        println("  $(rpad("definition",11)) | " * join([rpad(ic, 17) for ic in ICS_P], " "))
        for d in DEFS
            cells = String[]
            for ic in ICS_P
                r, f, ns = metrics(Ξs[(d, ic)], Ξtrue)
                push!(cells, rpad("$(round(r,digits=2))/$(round(f,digits=2))/$(round(ns,digits=1))", 17))
            end
            println("  $(rpad(d,11)) | " * join(cells, " "))
            flush(stdout)
        end
        println("  (each cell = recovery / FPR / n_selected)")
    end

    println("\nREAD: SLIC's row must be identical across all three definitions (it is")
    println("invariant to n_eff by construction). If the competitors' rows are also")
    println("stable, the conclusions do not depend on which n_eff definition is used,")
    println("which is what the response letter claims. If they move, report the")
    println("headline under the definition and state the sensitivity.")
end

case_idx = haskey(ENV, "SLURM_ARRAY_TASK_ID") ? parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) + 1 :
           (length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1)
main(case_idx)
