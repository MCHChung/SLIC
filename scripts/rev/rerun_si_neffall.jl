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

include(scriptsdir("rev", "sparse_regress_probe.jl"))  # score_on_all override; src/ untouched

# ============================================================================
# Fig. 3 recomputed with an effective sample size (Reviewer 2, §3).
#
# The reviewer asks: "I would ask the authors to recompute Figs. 2 and 3 with
# an effective sample size for the competitors. ... SLIC's penalty should be
# left untouched."
#
# Fig. 2 was done in rerun_with_neff_v2.jl. This script does Fig. 3's four
# panels:
#   ss  -> panel (c) subsampling         (dict_main, stride Δ ∈ SS_GRID)
#   Ts  -> panel (b) trajectory length   (dict_T,  one entry per length)
#   dts -> panel (a) sampling frequency  (dict_dt, one entry per dt)
#   ps  -> panel (d) ODE parameters      (dict_ps, own Ξtrue per parameter set)
#
# Each condition is scored three ways, matching neff_v2:
#   raw : n everywhere (as published)
#   p1  : n_eff in BOTH the fit coefficient and the penalty  (coherent)
#   p2  : n_eff in the penalty only
# SLIC is invariant to all three by construction (n cancels in the comparison
# n·log(k·RSS/n)); it is still scored under each as a control.
#
# Usage:
#   SLIC_SI_COND=ss  sbatch --array=0-35 cluster/submit_si_neff.sh
#   SLIC_SI_COND=Ts  sbatch --array=0-35 cluster/submit_si_neff.sh
#   SLIC_SI_COND=dts sbatch --array=0-35 cluster/submit_si_neff.sh
#   SLIC_SI_COND=ps  sbatch --array=0-23 cluster/submit_si_neff.sh   # sys ∈ [1,3,4,5]
# task = (sys_idx)*6 + noise_idx
# ============================================================================

const NOISE_SI   = [0, 5, 10, 20, 30, 40]
const RUNS_SI    = 25
const BATCHES_SI = 20
const TOL_SI     = 0.7
const ICS_SI     = ["slic", "aic", "aicc", "bic", "hqic", "bc", "kic"]

# Per-system conditioning weight, matching discover_model_ode_si.jl call sites
# (Lorenz c=1e-1 at line 1618; same values as the main benchmark).
const C_VALS_SI = Dict(1 => 1e-1, 2 => 1e-1, 3 => 1e-3, 4 => 1e-2, 5 => 1e0, 6 => 1e-2)

const SS_GRID = 1:3                      # matches DiscoverSIODEModel_ss default
const PS_SYSTEMS = [1, 3, 4, 5]          # ps driver asserts sys ∈ [1,3,4,5]

const SI_FILES = Dict(
    1 => "lordata_si.jld", 2 => "rossdata_si.jld", 3 => "lvdata_si.jld",
    4 => "brusdata_si.jld", 5 => "vdpdata_si.jld", 6 => "nlpdata_si.jld")

# ---------------------------------------------------------------------------
# GetInputsWithNeff does not accept Δ (the SI subsampling stride), and its
# n_eff ignores Δ entirely — it returns T_total·sqrt(p/π)/T_w regardless of how
# many windows were actually retained. With a large stride that can exceed the
# number of projected observations, which is meaningless. This variant:
#   (i)  threads Δ through to dwInt_sc / Lib, as the SI drivers do;
#   (ii) caps n_eff at the number of projected observations:
#            n_eff = min(n_proj, T_total·sqrt(p/π)/T_w)
#        For Δ·dt < T_w the windows still overlap and the kernel-limited value
#        binds (the cap is inactive); once Δ·dt ≥ T_w the windows separate and
#        each projected point is its own sample. For SS_GRID = 1:3 at the
#        benchmark dt the cap should never bind — which is itself the point of
#        panel (c): subsampling by 3 changes the NOMINAL n by 3× while leaving
#        the information content untouched.
# ---------------------------------------------------------------------------
function GetInputsWithNeffΔ(sys::Int, Xs, ts, Lib::Function, NoisePct; p=10, Δ=1)
    if sys in (1, 2, 3, 4, 7)
        Xtrue = Xs[1]
        η = NoisePct*mean(std(Xtrue, dims=2))/100
        Xn = Xtrue + η.*randn(size(Xtrue))
        Xsm = smooth_ode(Xn)
        _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)
        qt = dwInt_sc(Xsm, ts, wind, p=p, Δ=Δ)
        θ  = Lib(Xsm, ts, wind, p=p, Δ=Δ)

        dt = ts[2] - ts[1]
        T_total = ts[end] - ts[1]
        n_eff_traj = min(size(θ, 1), n_eff_gaussian(T_total, wind*dt; p=p))

        for i = 2:length(Xs)
            Xtrue = Xs[i]
            η = NoisePct*mean(std(Xtrue, dims=2))/100
            Xn = Xtrue + η.*randn(size(Xtrue))
            Xsm = smooth_ode(Xn)
            _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)
            qt_i = dwInt_sc(Xsm, ts, wind, p=p, Δ=Δ)
            θ_i  = Lib(Xsm, ts, wind, p=p, Δ=Δ)
            n_eff_traj += min(size(θ_i, 1), n_eff_gaussian(T_total, wind*dt; p=p))
            qt = hcat(qt, qt_i)
            θ  = vcat(θ, θ_i)
        end
        return qt, θ, n_eff_traj

    elseif sys == 5
        xtrue = Xs[1][1, :]
        η = NoisePct*std(xtrue)/100
        xn = xtrue + η.*randn(size(xtrue))
        xsm = smooth_ode(xn)
        ts_tr, x, dx = DataWithFirstDeriv(xsm', ts, ts[2]-ts[1])
        Xsm = vcat(x, dx)
        _,_,wind = FindW(Xsm, ts_tr, ws=21:2:121, p=p)
        qt = dwInt_sc(Xsm, ts_tr, wind, p=p, Δ=Δ)
        θ  = Lib(Xsm, ts_tr, wind, p=p, Δ=Δ)

        dt = ts[2] - ts[1]
        T_total = ts_tr[end] - ts_tr[1]
        n_eff_traj = min(size(θ, 1), n_eff_gaussian(T_total, wind*dt; p=p))

        for i = 2:length(Xs)
            xtrue = Xs[i][1, :]
            η = NoisePct*std(xtrue)/100
            xn = xtrue + η.*randn(size(xtrue))
            xsm = smooth_ode(xn)
            ts_tr, x, dx = DataWithFirstDeriv(xsm', ts, ts[2]-ts[1])
            Xsm = vcat(x, dx)
            _,_,wind = FindW(Xsm, ts_tr, ws=21:2:121, p=p)
            qt_i = dwInt_sc(Xsm, ts_tr, wind, p=p, Δ=Δ)
            θ_i  = Lib(Xsm, ts_tr, wind, p=p, Δ=Δ)
            n_eff_traj += min(size(θ_i, 1), n_eff_gaussian(T_total, wind*dt; p=p))
            qt = hcat(qt, qt_i)
            θ  = vcat(θ, θ_i)
        end
        return qt, θ, n_eff_traj

    elseif sys == 6
        # NLP: 1-state, SECOND-order weak form (ẍ = -ω² sin x). Mirrors
        # GetInputsWithNeff's sys==6 branch exactly, including the (1, n_obs)
        # reshape EnAdSR requires. NOTE: uses d2wInt_sc, not dwInt_sc, and
        # reshape(xsm) rather than DataWithFirstDeriv — collapsing this into
        # the sys==5 branch is what broke task 35.
        xtrue = Xs[1][1, :]
        η = NoisePct*std(xtrue)/100
        xn = xtrue + η.*randn(size(xtrue))
        xsm = smooth_ode(xn)
        Xsm = reshape(xsm, (1, length(xsm)))
        _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)
        qt = d2wInt_sc(Xsm, ts, wind, p=p, Δ=Δ)
        qt = reshape(qt, (1, length(qt)))
        θ  = Lib(Xsm, ts, wind, p=p, Δ=Δ)

        dt = ts[2] - ts[1]
        T_total = ts[end] - ts[1]
        n_eff_traj = min(size(θ, 1), n_eff_gaussian(T_total, wind*dt; p=p))

        for i = 2:length(Xs)
            xtrue = Xs[i][1, :]
            η = NoisePct*std(xtrue)/100
            xn = xtrue + η.*randn(size(xtrue))
            xsm = smooth_ode(xn)
            Xsm = reshape(xsm, (1, length(xsm)))
            _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)
            qt_i = d2wInt_sc(Xsm, ts, wind, p=p, Δ=Δ)
            qt_i = reshape(qt_i, (1, length(qt_i)))
            θ_i  = Lib(Xsm, ts, wind, p=p, Δ=Δ)
            n_eff_traj += min(size(θ_i, 1), n_eff_gaussian(T_total, wind*dt; p=p))
            qt = hcat(qt, qt_i)
            θ  = vcat(θ, θ_i)
        end
        return qt, θ, n_eff_traj
    else
        error("unsupported sys=$sys")
    end
end

"""Return (list of (Xs, ts, Δ, Ξtrue, label)) for the requested condition."""
function sweep_levels(si_cond::String, sys::Int)
    data = load(datadir("sims", "ode_data", SI_FILES[sys]))
    out = []
    if si_cond == "ss"
        d = data["dict_main"]
        for Δ in SS_GRID
            push!(out, (d["Xtrues"], d["ts"], Δ, d["Ξtrue"], "Δ=$Δ"))
        end
    elseif si_cond == "Ts"
        d = data["dict_T"]
        for i = 1:length(d["ts"])
            push!(out, (d["Xtrues"][i], d["ts"][i], 1, d["Ξtrue"], "T#$i"))
        end
    elseif si_cond == "dts"
        d = data["dict_dt"]
        for i = 1:length(d["ts"])
            push!(out, (d["Xtrues"][i], d["ts"][i], 1, d["Ξtrue"], "dt#$i"))
        end
    elseif si_cond == "ps"
        d = data["dict_ps"]
        for i = 1:length(d["Xtrues"])
            push!(out, (d["Xtrues"][i], d["ts"], 1, d["Ξtrues"][i], "ps#$i"))
        end
    else
        error("cond must be ss | Ts | dts | ps")
    end
    return out
end

function main(si_cond::String, sys::Int, noise_idx::Int)
    @assert 1 <= noise_idx <= length(NOISE_SI)
    NoisePct = NOISE_SI[noise_idx]
    _, sysname, n_state, _, _, _ = SYSTEMS[sys]
    Lib   = LIBS[sys]
    c_sys = C_VALS_SI[sys]

    levels = sweep_levels(si_cond, sys)
    nlev = length(levels)

    println("=== Fig3 n_eff | si_cond=$si_cond | $sysname (sys=$sys) | noise=$NoisePct% | $nlev levels ===")
    flush(stdout)

    n_lib = size(levels[1][4], 1)
    Ξs = Dict("neffall" => Dict(ic => zeros(n_lib, n_state, RUNS_SI, nlev)
                                for ic in ICS_SI))
    n_eff_rec = zeros(RUNS_SI, nlev)
    Ξtrues    = [levels[l][4] for l = 1:nlev]

    for l = 1:nlev
        Xs, ts_l, Δ, _, label = levels[l]
        println("  level $l/$nlev ($label)"); flush(stdout)
        for run = 1:RUNS_SI
            if run % 5 == 1; println("    run $run/$RUNS_SI"); flush(stdout); end
            Random.seed!(7000 + 131*sys + 17*noise_idx + 3*l + run)
            qt, θ, n_eff_val = GetInputsWithNeffΔ(sys, Xs, ts_l, Lib, NoisePct; Δ=Δ)
            n_eff_rec[run, l] = n_eff_val
            for ic in ICS_SI
                # neff_all: n_eff in fit AND penalty, scored on all n (score_on_all=true)
                Ξ, _, _ = EnAdSR(θ, qt', ic; tol=TOL_SI, c=c_sys, num_batches=BATCHES_SI,
                                 n_eff=n_eff_val, fit_uses_neff=true, score_on_all=true)
                Ξs["neffall"][ic][:, :, run, l] = Ξ
            end
        end
    end

    outdir = datadir("sims", "ode_results_rev", "si_neffall", si_cond)
    mkpath(outdir)
    sysname_safe = lowercase(replace(sysname, " " => "_"))
    payload = Dict{String,Any}(
        "sys" => sys, "sysname" => sysname, "cond" => si_cond,
        "NoisePct" => NoisePct, "noise_idx" => noise_idx,
        "n_levels" => nlev, "labels" => [levels[l][5] for l = 1:nlev],
        "Xitrues" => Ξtrues, "n_eff" => n_eff_rec, "runs" => RUNS_SI,
        "c" => c_sys)
    for ic in ICS_SI
        payload["Xis_neffall_$(ic)"] = Ξs["neffall"][ic]
    end
    wsave(joinpath(outdir, "$(sysname_safe)_si$(si_cond)_noise$(noise_idx).jld"), payload)
    println("  wrote $(outdir)")
end

si_cond = ENV["SLIC_SI_COND"]
task = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
syslist = si_cond == "ps" ? PS_SYSTEMS : [1, 2, 3, 4, 5, 6]
sys       = syslist[(task ÷ length(NOISE_SI)) + 1]
noise_idx = (task % length(NOISE_SI)) + 1
main(si_cond, sys, noise_idx)
