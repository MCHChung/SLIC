using LinearAlgebra

# ============================================================================
# No-projection diagnostic pipeline (R2 §3 complementary experiment).
#
# REVISION (v2): switched from 2nd-order centered FD to:
#   - `CalcDeriv` from derivative.jl (4th-order centered) for first derivatives
#   - A 4th-order central stencil for second derivatives, rather than
#     cascading two first derivatives (which compounds truncation error
#     and amplifies noise).
#
# Stencils:
#   d/dt u[i]  = (-u[i+2] + 8 u[i+1] - 8 u[i-1] + u[i-2]) / (12 dt)     [O(dt^4)]
#   d²/dt² u[i] = (-u[i+2] + 16 u[i+1] - 30 u[i] + 16 u[i-1] - u[i-2]) / (12 dt²) [O(dt^4)]
#
# The endpoints (first 2 and last 2 samples) are dropped, not extrapolated.
# ============================================================================

include("derivative.jl")  # CalcDeriv, DataWithFirstDeriv
include("smooth.jl")      # smooth_ode (Whittaker-Henderson with GCV)

# 4th-order centered second derivative
function _second_deriv_4th(u::AbstractVector, dt::Real)
    n = length(u)
    @assert n >= 5 "Need at least 5 points for 4th-order second derivative"
    d2u = zeros(n - 4)
    for i = 3:(n - 2)
        d2u[i - 2] = (-u[i+2] + 16*u[i+1] - 30*u[i] + 16*u[i-1] - u[i-2]) / (12 * dt^2)
    end
    return d2u
end

"""
    GetInputsNoProj(sys, Xs, ts, Lib, NoisePct; subsample=10, p=10, assume_velocity_known=false)

Build (qt, θ) for no-projection diagnostic using 4th-order centered finite
differences. The pipeline:
  1. Add noise to ground-truth trajectories
  2. Smooth the noisy data with smooth_ode (Whittaker-Henderson + GCV)
  3. Estimate dx/dt via CalcDeriv (4th-order); d²x/dt² via the 4th-order
     central stencil (NOT by cascading two first derivatives)
  4. Subsample by factor `subsample` to reduce data size and decorrelate
  5. Evaluate library terms pointwise on the smoothed signal

Returns (qt, θ) where qt has shape (n_state, n_obs) and θ is (n_obs, n_lib).

For VdP (sys=5) and Pendulum (sys=6):
  - `assume_velocity_known=false` (default): only x observed; v estimated
    from CalcDeriv. For VdP, this means CalcDeriv for v then for a (still
    1 numerical derivative each but each is 4th-order).
  - `assume_velocity_known=true` (VdP only): both x,v observed and noised.
    Only one numerical differentiation needed (v -> a).
"""
function GetInputsNoProj(sys::Int, Xs, ts, Lib::Function, NoisePct;
                          subsample::Int=10, p::Int=10,
                          assume_velocity_known::Bool=false)
    dt = ts[2] - ts[1]
    if sys == 1 || sys == 2 || sys == 3 || sys == 4 || sys == 7
        # Fully observed: use CalcDeriv pointwise on smoothed signal
        return _fully_observed_no_proj(sys, Xs, ts, dt, NoisePct, subsample)

    elseif sys == 5  # Van der Pol
        if assume_velocity_known
            return _vdp_with_known_velocity_v2(Xs, ts, dt, NoisePct, subsample)
        else
            return _vdp_estimate_velocity_v2(Xs, ts, dt, NoisePct, subsample)
        end

    elseif sys == 6  # Nonlinear pendulum: directly compute d²x/dt² with 4th-order stencil
        return _pendulum_no_proj_v2(Xs, ts, dt, NoisePct, subsample)
    end
end

# ============================================================================
# Fully observed systems (Lorenz, Rossler, LV, Brusselator, Duffing)
# ============================================================================

function _fully_observed_no_proj(sys, Xs, ts, dt, NoisePct, subsample)
    qt_full = nothing
    θ_full = nothing

    for j = 1:length(Xs)
        Xtrue = Xs[j]
        η = NoisePct * mean(std(Xtrue, dims=2)) / 100
        Xn = Xtrue + η .* randn(size(Xtrue))
        Xsm = smooth_ode(Xn)                # shape (n_state, n_t)
        dXdt = CalcDeriv(Xsm', dt)'          # shape (n_state, n_t - 4)
        idx_full = 3:(size(Xsm, 2) - 2)      # time indices corresponding to dXdt rows
        Xsm_trim = Xsm[:, idx_full]
        sub = 1:subsample:size(dXdt, 2)
        qt_j = dXdt[:, sub]
        θ_j = _build_library_pointwise(sys, Xsm_trim[:, sub])
        if isnothing(qt_full)
            qt_full = qt_j; θ_full = θ_j
        else
            qt_full = hcat(qt_full, qt_j)
            θ_full = vcat(θ_full, θ_j)
        end
    end
    return qt_full, θ_full
end

# ============================================================================
# Van der Pol — two conditions
# ============================================================================

# estimate_v: only x observed; v and a both estimated from CalcDeriv
function _vdp_estimate_velocity_v2(Xs, ts, dt, NoisePct, subsample)
    qt_full = nothing; θ_full = nothing
    for j = 1:length(Xs)
        xtrue = Xs[j][1, :]
        η = NoisePct * std(xtrue) / 100
        xn = xtrue + η .* randn(size(xtrue))
        xsm = vec(smooth_ode(xn))    # smooth_ode returns a Matrix; flatten to Vector
        n_t = length(xsm)
        # v from CalcDeriv (first derivative, 4th-order)
        v = vec(CalcDeriv(reshape(xsm, (n_t, 1)), dt))   # length n_t - 4
        # a from the direct 4th-order second-derivative stencil (single pass,
        # avoids compounding error from cascading two first derivatives)
        a_full = _second_deriv_4th(xsm, dt)   # length n_t - 4
        # Both v and a are now length n_t - 4, on time indices 3:n_t-2
        # Trim x to match
        x_trim = xsm[3:end-2]
        Xsm_trim = vcat(reshape(x_trim, (1, length(x_trim))),
                        reshape(v, (1, length(v))))
        sub = 1:subsample:length(v)
        qt_j = hcat(v[sub], a_full[sub])'   # (2, n_sub)
        θ_j = _build_library_pointwise(5, Xsm_trim[:, sub])
        if isnothing(qt_full)
            qt_full = qt_j; θ_full = θ_j
        else
            qt_full = hcat(qt_full, qt_j)
            θ_full = vcat(θ_full, θ_j)
        end
    end
    return qt_full, θ_full
end

# known_v: both x and v observed (noised independently); a from CalcDeriv on v
function _vdp_with_known_velocity_v2(Xs, ts, dt, NoisePct, subsample)
    qt_full = nothing; θ_full = nothing
    for j = 1:length(Xs)
        X = Xs[j]
        xtrue = X[1, :]; vtrue = X[2, :]
        ηx = NoisePct * std(xtrue) / 100
        ηv = NoisePct * std(vtrue) / 100
        xn = xtrue + ηx .* randn(size(xtrue))
        vn = vtrue + ηv .* randn(size(vtrue))
        xsm = vec(smooth_ode(xn)); vsm = vec(smooth_ode(vn))
        n_t = length(xsm)
        # a = dv/dt via CalcDeriv
        a = vec(CalcDeriv(reshape(vsm, (n_t, 1)), dt))   # length n_t - 4
        # Trim x and v
        x_trim = xsm[3:end-2]; v_trim = vsm[3:end-2]
        Xsm_trim = vcat(reshape(x_trim, (1, length(x_trim))),
                        reshape(v_trim, (1, length(v_trim))))
        sub = 1:subsample:length(a)
        # qt: (v, a) — note v is the FIRST derivative, given as data, not estimated
        qt_j = hcat(v_trim[sub], a[sub])'
        θ_j = _build_library_pointwise(5, Xsm_trim[:, sub])
        if isnothing(qt_full)
            qt_full = qt_j; θ_full = θ_j
        else
            qt_full = hcat(qt_full, qt_j)
            θ_full = vcat(θ_full, θ_j)
        end
    end
    return qt_full, θ_full
end

# ============================================================================
# Pendulum (sys=6) — 4th-order second-derivative stencil directly
# ============================================================================

function _pendulum_no_proj_v2(Xs, ts, dt, NoisePct, subsample)
    qt_full = nothing; θ_full = nothing
    for j = 1:length(Xs)
        xtrue = Xs[j][1, :]
        η = NoisePct * std(xtrue) / 100
        xn = xtrue + η .* randn(size(xtrue))
        xsm = vec(smooth_ode(xn))    # smooth_ode returns a Matrix; flatten to Vector
        a_full = _second_deriv_4th(xsm, dt)
        x_trim = xsm[3:end-2]
        Xsm_trim = reshape(x_trim, (1, length(x_trim)))
        sub = 1:subsample:length(a_full)
        qt_j = reshape(a_full[sub], (1, length(sub)))   # (1, n_sub)
        θ_j = _build_library_pointwise(6, Xsm_trim[:, sub])
        if isnothing(qt_full)
            qt_full = qt_j; θ_full = θ_j
        else
            qt_full = hcat(qt_full, qt_j)
            θ_full = vcat(θ_full, θ_j)
        end
    end
    return qt_full, θ_full
end

# ============================================================================
# Library builders: pointwise versions (unchanged from v1)
# ============================================================================

function _build_library_pointwise(sys::Int, X)
    n_state, n_t = size(X)
    if sys == 1 || sys == 3 || sys == 5 || sys == 7  # polynomial degree 3, no bias
        θ = reshape(X[1, :], (n_t, 1))
        for i = 2:n_state
            θ = hcat(θ, X[i, :])
        end
        for i = 1:n_state, j = i:n_state
            θ = hcat(θ, X[i, :] .* X[j, :])
        end
        for i = 1:n_state, j = i:n_state, k = j:n_state
            θ = hcat(θ, X[i, :] .* X[j, :] .* X[k, :])
        end
        return θ
    elseif sys == 2 || sys == 4   # polynomial degree 3 with bias term
        θ = reshape(ones(n_t), (n_t, 1))
        for i = 1:n_state
            θ = hcat(θ, X[i, :])
        end
        for i = 1:n_state, j = i:n_state
            θ = hcat(θ, X[i, :] .* X[j, :])
        end
        for i = 1:n_state, j = i:n_state, k = j:n_state
            θ = hcat(θ, X[i, :] .* X[j, :] .* X[k, :])
        end
        return θ
    elseif sys == 6  # Fourier basis sin/cos to order 5
        θ = reshape(ones(n_t), (n_t, 1))
        for i = 1:5
            θ = hcat(θ, sin.(i * X[1, :]))
            θ = hcat(θ, cos.(i * X[1, :]))
        end
        return θ
    end
end
