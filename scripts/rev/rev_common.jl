using DrWatson
@quickactivate "SLIC"
using JLD, Random, Statistics

# ============================================================================
# Common helpers for revision scripts. Pulls in modified pipeline pieces.
# ============================================================================

include(srcdir("sparse_regress.jl"))    # MODIFIED with n_eff, WAIC, NML
include(srcdir("derivative.jl"))
include(srcdir("galerkin_proj.jl"))
include(srcdir("smooth.jl"))
include(srcdir("vis_results.jl"))
include(srcdir("neff.jl"))               # NEW
include(srcdir("enumeration.jl"))        # NEW
include(srcdir("no_projection.jl"))      # NEW

# ----------------------------------------------------------------------------
# System metadata used across rev scripts
# ----------------------------------------------------------------------------

const SYSTEMS = Dict(
    1 => ("lordata.jld",  "Lorenz",         3, 19, 0.001, 10.0),
    2 => ("rossdata.jld", "Rossler",        3, 20, 0.001, 10.0),
    3 => ("lvdata.jld",   "Lotka-Volterra", 2,  9, 0.01,  20.0),
    4 => ("brusdata.jld", "Brusselator",    2, 10, 0.001, 10.0),
    5 => ("vdpdata.jld",  "Van der Pol",    2,  9, 0.01,  30.0),
    6 => ("nlpdata.jld",  "Nonlin Pendulum",1, 11, 0.01,  20.0),
)

# Library functions for each system

function LorLib(X, ts, wind; p=10, Δ=1)
    θ = wInt_sc(X[1,:], ts, wind, p=p,Δ=Δ)
    for i=2:size(X,1)
        θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
    end
    for i=1:size(X,1), j=i:size(X,1)
        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
    end
    for i=1:size(X,1), j=i:size(X,1), k=j:size(X,1)
        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
    end
    return θ
end

function RossLib(X, ts, wind; p=10, Δ=1)
    θ = wInt_sc(ones(size(X[1,:])), ts, wind, p=p,Δ=Δ)
    for i=1:size(X,1)
        θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
    end
    for i=1:size(X,1), j=i:size(X,1)
        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
    end
    for i=1:size(X,1), j=i:size(X,1), k=j:size(X,1)
        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
    end
    return θ
end

function LVLib(X, ts, wind; p=10, Δ=1)
    θ = wInt_sc(X[1,:], ts, wind, p=p,Δ=Δ)
    for i=2:size(X,1)
        θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
    end
    for i=1:size(X,1), j=i:size(X,1)
        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
    end
    for i=1:size(X,1), j=i:size(X,1), k=j:size(X,1)
        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
    end
    return θ
end

function BrusLib(X, ts, wind; p=10, Δ=1)
    θ = wInt_sc(ones(size(X[1,:])), ts, wind, p=p,Δ=Δ)
    for i=1:size(X,1)
        θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
    end
    for i=1:size(X,1), j=i:size(X,1)
        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
    end
    for i=1:size(X,1), j=i:size(X,1), k=j:size(X,1)
        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
    end
    return θ
end

function VdPLib(X, ts, wind; p=10, Δ=1)
    θ = wInt_sc(X[1,:], ts, wind, p=p,Δ=Δ)
    for i=2:size(X,1)
        θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
    end
    for i=1:size(X,1), j=i:size(X,1)
        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
    end
    for i=1:size(X,1), j=i:size(X,1), k=j:size(X,1)
        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
    end
    return θ
end

function NLPLib(X, ts, wind; p=10, Δ=1)
    θ = wInt_sc(ones(length(X[:])), ts, wind, p=p, Δ=Δ)
    for i=1:5
        θ = hcat(θ, wInt_sc(sin.(i*X[:]), ts, wind, p=p, Δ=Δ))
        θ = hcat(θ, wInt_sc(cos.(i*X[:]), ts, wind, p=p, Δ=Δ))
    end
    return θ
end

# Duffing library (REVISION: R2 §4): same structure as VdP but tracks cubic
function DuffLib(X, ts, wind; p=10, Δ=1)
    θ = wInt_sc(X[1,:], ts, wind, p=p,Δ=Δ)
    for i=2:size(X,1)
        θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
    end
    for i=1:size(X,1), j=i:size(X,1)
        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
    end
    for i=1:size(X,1), j=i:size(X,1), k=j:size(X,1)
        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
    end
    return θ
end

const LIBS = Dict(
    1 => LorLib,  2 => RossLib, 3 => LVLib,
    4 => BrusLib, 5 => VdPLib,  6 => NLPLib,
    7 => DuffLib,
)

# ----------------------------------------------------------------------------
# GetInputs that also returns n_eff (Gaussian-equivalent) alongside qt, θ.
# Convention: qt has shape (n_state, n_obs). Scripts uniformly call EnAdSR
# with qt' (giving (n_obs, n_state)).
# ----------------------------------------------------------------------------

function GetInputsWithNeff(sys::Int, Xs, ts, Lib::Function, NoisePct; p=10)
    if sys==1 || sys==2 || sys==3 || sys==4 || sys==7
        Xtrue = Xs[1]
        η = NoisePct*mean(std(Xtrue, dims=2))/100
        Xn = Xtrue + η.*randn(size(Xtrue))
        Xsm = smooth_ode(Xn)
        _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)
        qt = dwInt_sc(Xsm, ts, wind, p=p)    # (n_state, n_obs)
        θ = Lib(Xsm, ts, wind, p=p)

        dt = ts[2] - ts[1]
        Tw = wind * dt
        T_total = ts[end] - ts[1]
        n_eff_traj = n_eff_gaussian(T_total, Tw; p=p)

        for i=2:length(Xs)
            Xtrue = Xs[i]
            η = NoisePct*mean(std(Xtrue, dims=2))/100
            Xn = Xtrue + η.*randn(size(Xtrue))
            Xsm = smooth_ode(Xn)
            _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)
            qt_i = dwInt_sc(Xsm, ts, wind, p=p)
            θ_i = Lib(Xsm, ts, wind, p=p)
            Tw = wind * dt
            n_eff_traj += n_eff_gaussian(T_total, Tw; p=p)
            qt = hcat(qt, qt_i)
            θ = vcat(θ, θ_i)
        end
        return qt, θ, n_eff_traj

    elseif sys==5
        xtrue = Xs[1][1,:]
        η = NoisePct*std(xtrue)/100
        xn = xtrue + η.*randn(size(xtrue))
        xsm = smooth_ode(xn)
        ts_tr, x, dx = DataWithFirstDeriv(xsm', ts, ts[2]-ts[1])
        Xsm = vcat(x, dx)
        _,_,wind = FindW(Xsm, ts_tr, ws=21:2:121, p=p)
        qt = dwInt_sc(Xsm, ts_tr, wind, p=p)
        θ = Lib(Xsm, ts_tr, wind, p=p)

        dt = ts[2] - ts[1]
        Tw = wind * dt
        T_total = ts_tr[end] - ts_tr[1]
        n_eff_traj = n_eff_gaussian(T_total, Tw; p=p)

        for i=2:length(Xs)
            xtrue = Xs[i][1,:]
            η = NoisePct*std(xtrue)/100
            xn = xtrue + η.*randn(size(xtrue))
            xsm = smooth_ode(xn)
            ts_tr, x, dx = DataWithFirstDeriv(xsm', ts, ts[2]-ts[1])
            Xsm = vcat(x, dx)
            _,_,wind = FindW(Xsm, ts_tr, ws=21:2:121, p=p)
            qt_i = dwInt_sc(Xsm, ts_tr, wind, p=p)
            θ_i = Lib(Xsm, ts_tr, wind, p=p)
            Tw = wind * dt
            n_eff_traj += n_eff_gaussian(T_total, Tw; p=p)
            qt = hcat(qt, qt_i)
            θ = vcat(θ, θ_i)
        end
        return qt, θ, n_eff_traj

    elseif sys==6
        # FIX: previously qt was reshaped to (n_obs, 1), which made the
        # uniform `qt'` in the experiment scripts produce (1, n_obs) — wrong
        # orientation for EnAdSR, causing cond(θ_train) to fail on an empty
        # matrix when 80% of "1 observation" was sampled.
        # Now qt is (1, n_obs), matching sys=1-5 convention.
        xtrue = Xs[1][1,:]
        η = NoisePct*std(xtrue)/100
        xn = xtrue + η.*randn(size(xtrue))
        xsm = smooth_ode(xn)
        Xsm = reshape(xsm, (1, length(xsm)))
        _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)
        qt = d2wInt_sc(Xsm, ts, wind, p=p)
        qt = reshape(qt, (1, length(qt)))     # (n_state=1, n_obs)
        θ = Lib(Xsm, ts, wind, p=p)

        dt = ts[2] - ts[1]
        Tw = wind * dt
        T_total = ts[end] - ts[1]
        n_eff_traj = n_eff_gaussian(T_total, Tw; p=p)

        for i=2:length(Xs)
            xtrue = Xs[i][1,:]
            η = NoisePct*std(xtrue)/100
            xn = xtrue + η.*randn(size(xtrue))
            xsm = smooth_ode(xn)
            Xsm = reshape(xsm, (1, length(xsm)))
            _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)
            qt_i = d2wInt_sc(Xsm, ts, wind, p=p)
            qt_i = reshape(qt_i, (1, length(qt_i)))
            θ_i = Lib(Xsm, ts, wind, p=p)
            Tw = wind * dt
            n_eff_traj += n_eff_gaussian(T_total, Tw; p=p)
            # Concatenate observations: hcat along dim 2 for qt, vcat for θ
            qt = hcat(qt, qt_i)
            θ = vcat(θ, θ_i)
        end
        return qt, θ, n_eff_traj
    end
end
