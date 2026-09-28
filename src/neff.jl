using Statistics, LinearAlgebra

# ============================================================================
# Effective sample size computation for Galerkin-projected observations.
# Implements three definitions, in increasing rigor:
#
#   1. Simple proxy:           n_eff = T_total / T_w
#                              (treats kernel as uniform window of width T_w)
#
#   2. Gaussian-equivalent:    n_eff = T_total * sqrt(p/π) / T_w
#                              (accounts for peaked kernel w(u) ∝ (1-u²)^p ≈ exp(-p u²);
#                              this is the autocorrelation-equivalent uniform width
#                              T_w^eff = T_w * sqrt(π/p))
#
#   3. Empirical autocorrelation: n_eff = n_proj / τ_eff
#                              (τ_eff = 1 + 2 Σ ρ(ℓ) from residual autocorrelation,
#                              with Sokal's automatic windowing rule for cutoff)
#
# See the Supplementary Information for the derivation of (2).
# ============================================================================

"""
    n_eff_simple(T_total, T_w)

Simple proxy for effective sample size. Treats the kernel as a uniform
window of width `T_w`. Returns `T_total / T_w`.

Used as a baseline; not recommended as the headline value for the peaked
polynomial kernel.
"""
function n_eff_simple(T_total::Real, T_w::Real)
    return T_total / T_w
end

"""
    n_eff_gaussian(T_total, T_w; p=10)

Gaussian-equivalent effective sample size for the polynomial kernel
w(u) ∝ (1 - u²)^p. Approximates the kernel as exp(-p u²) (well-justified for
p ≥ 5; see SI for kernel-shape comparison) and computes the autocorrelation-
equivalent uniform-window width T_w^eff = T_w · sqrt(π/p), giving:

    n_eff = T_total · sqrt(p/π) / T_w

Recommended as the headline n_eff value for the manuscript.
"""
function n_eff_gaussian(T_total::Real, T_w::Real; p::Int=10)
    return T_total * sqrt(p / π) / T_w
end

"""
    n_eff_autocorr(residuals; max_lag=nothing)

Empirical effective sample size from residual autocorrelation.

`residuals` is a vector (or matrix; columns treated as separate trajectories).
Computes the integrated autocorrelation time τ_eff = 1 + 2 Σ ρ(ℓ) with
Sokal's automatic windowing rule (smallest M such that M ≥ 5·τ_eff(M))
to determine the truncation cutoff.

Returns n_eff = n / τ_eff.
"""
function n_eff_autocorr(residuals::AbstractVector; max_lag=nothing)
    n = length(residuals)
    r = residuals .- mean(residuals)
    var_r = var(r)
    if var_r < eps()
        return float(n)  # zero variance: treat as fully independent
    end

    M_default = isnothing(max_lag) ? min(n ÷ 4, 1000) : max_lag

    # Autocorrelation via direct sum (acceptable for n up to ~10^5;
    # for larger n switch to FFT-based)
    ρ = zeros(M_default)
    for ℓ = 1:M_default
        ρ[ℓ] = sum(r[1:end-ℓ] .* r[ℓ+1:end]) / ((n - ℓ) * var_r)
    end

    # Sokal's windowing rule: smallest M with M ≥ 5·τ(M)
    τ = 1.0
    M = M_default
    for k = 1:M_default
        τ += 2 * ρ[k]
        if k ≥ 5 * τ
            M = k
            break
        end
    end

    τ = max(τ, 1.0)
    return n / τ
end

function n_eff_autocorr(residuals::AbstractMatrix; kwargs...)
    # If residuals is a matrix (n × n_state), average n_eff over state dims
    n_states = size(residuals, 2)
    return mean(n_eff_autocorr(residuals[:, s]; kwargs...) for s = 1:n_states)
end

"""
    n_eff_trace(P)

Effective sample size from projection-operator trace (rigorous version).
`P` is the sparse projection matrix (n_proj × n_raw) that maps raw signal
to projected observations. Returns tr(PᵀP) under row-sum-to-one
normalization.

Equivalent to n_eff_gaussian in the limit of large p; can be computed
directly from the kernel weights without constructing the full P matrix.
"""
function n_eff_trace(weights_per_window::AbstractVector, n_proj::Int)
    # weights_per_window: kernel values w_j for one window (length = window size)
    # Per-window factor: Σ w_j² / (Σ w_j)²
    w_sum = sum(weights_per_window)
    w_sum_sq = sum(abs2, weights_per_window)
    per_window_factor = w_sum_sq / w_sum^2
    return n_proj * per_window_factor
end

"""
    polynomial_kernel_weights(wind::Int; p::Int=10)

Returns the polynomial kernel weights w_j = [(t_2-t_j)(t_j-t_1)]^p for
`wind` evenly spaced points in a window of width 1.
"""
function polynomial_kernel_weights(wind::Int; p::Int=10)
    u = range(-1.0, 1.0, length=wind)
    return [(1 - x^2)^p for x in u]
end
