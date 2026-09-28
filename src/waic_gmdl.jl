using LinearAlgebra, Random

# ============================================================================
# WAIC and gMDL for Gaussian linear regression, rewritten to be scored on the
# SAME footing as the classical criteria in src/sparse_regress.jl:score.
#
# Reference (score, src/sparse_regress.jl:129):
#     n     = size(y, 1)
#     n_pen = isnothing(n_eff) ? n : n_eff
#     n_fit = (isnothing(n_eff) || !fit_uses_neff) ? n : n_pen
#     k     = count(abs.(Ξ) .> 0.) + 1
#     RSS   = sum(abs2, y - θ*Ξ) + η
#     bic   = n_fit*log(RSS/n) + k*log(n_pen)          # etc.
#
# Four mismatches in the previous implementations, all fixed here:
#
#  1. UNION SUPPORT + REFIT (the serious one).  Both used
#         biginds = vec(any(abs.(Ξ) .> 0, dims=2))
#     i.e. the UNION of supports across equations, then refit EVERY equation on
#     ALL union columns. For Lorenz's true Ξ (7 nonzeros, 5 union columns) that
#     scores a 15-parameter model, not the 7-parameter one proposed. Worse, a
#     column added to ANY ONE equation becomes available to ALL equations on
#     refit, so the fit improves far more than the proposal warrants — a direct
#     reward for growing the union, and a plausible mechanism for the observed
#     FPR ~ 1. `score` instead evaluates the residual of the ACTUAL sparse Ξ.
#     Fixed: residual is y - θ*Ξ, no refit, sparsity pattern respected.
#
#  2. n_eff IGNORED.  Both accepted n_eff and never used it (bodies ran on
#     n_obs = size(X,1) throughout), so every n_eff-corrected WAIC/gMDL result
#     was a raw-n result. Fixed: n_fit / n_pen convention, matching score.
#
#  3. η NEVER APPLIED.  Both discarded the passed RSS (which carries + η) by
#     refitting. The five classical criteria all receive the floor. Fixed: the
#     passed, floored RSS is used.
#
#  4. k CONVENTION.  gMDL charged itself k_act = union columns while the others
#     are charged count(nonzeros)+1. Fixed for gMDL.
#     WAIC keeps pWAIC as its complexity term — pWAIC IS WAIC's own effective-
#     parameter estimate, and replacing it with count+1 would not be WAIC.
#
# Note on scale: each criterion is only ever minimised against itself across
# candidates, so overall factors (e.g. score's factor-2 convention) do not
# affect selection and are not matched.
# ============================================================================

"""
    waic_score(y, θ, Ξ, RSS, n; n_eff=nothing, n_samples=1000, rng=GLOBAL_RNG)

WAIC via exact draws from the conjugate posterior, scored on the sparse Ξ.

    σ² | y ~ Inv-Gamma((n_pen - k)/2, RSS/2)      RSS carries the η floor
    β  | σ² ~ N(Ξ[supp,s], σ² (Xₛᵀ Xₛ)⁻¹)          per equation, own support

    WAIC = -2 (lppd - pWAIC)

lppd is scaled by n_fit/n, mirroring score's `n_fit*log(RSS/n)` fit term.
Lower is better. `RSS` is the caller's floored total (do NOT recompute it).
"""
function waic_score(y, θ, Ξ, RSS, n; n_eff=nothing, n_samples=1000,
                    rng=Random.GLOBAL_RNG, fit_uses_neff::Bool=true)
    if !any(abs.(Ξ) .> 0)
        return Inf
    end

    n_obs   = size(y, 1)
    n_state = size(y, 2)
    n_pen   = isnothing(n_eff) ? float(n_obs) : float(n_eff)
    n_fit   = (isnothing(n_eff) || !fit_uses_neff) ? float(n_obs) : n_pen
    k       = count(abs.(Ξ) .> 0.) + 1

    if k >= n_pen
        return Inf
    end

    # One shared σ² posterior for the system, from the caller's floored RSS,
    # matching score's single-RSS / single-k treatment.
    shape = (n_pen - k) / 2
    scale = max(RSS, eps()) / 2
    if shape <= 0
        return Inf
    end

    # Per-equation design and Cholesky on that equation's OWN support.
    Xs      = Vector{Matrix{Float64}}(undef, n_state)
    Ls      = Vector{Matrix{Float64}}(undef, n_state)
    betas   = Vector{Vector{Float64}}(undef, n_state)
    for s = 1:n_state
        supp = vec(abs.(Ξ[:, s]) .> 0)
        if !any(supp)
            Xs[s] = zeros(n_obs, 0); Ls[s] = zeros(0, 0); betas[s] = Float64[]
            continue
        end
        X = θ[:, supp]
        Xs[s] = X
        betas[s] = Ξ[supp, s]            # already the OLS fit on this support
        XtX_inv = inv(Symmetric(X' * X))
        local L
        try
            L = cholesky(Symmetric(XtX_inv)).L
        catch
            L = cholesky(Symmetric(XtX_inv) + 1e-10 * I).L
        end
        Ls[s] = Matrix(L)
    end

    total_lppd  = 0.0
    total_pwaic = 0.0

    for s = 1:n_state
        ks = length(betas[s])
        loglik = Matrix{Float64}(undef, n_samples, n_obs)
        for j = 1:n_samples
            σ² = 1 / _rand_gamma(rng, shape, 1 / scale)
            σ  = sqrt(σ²)
            if ks == 0
                r = y[:, s]
            else
                β = betas[s] + σ * (Ls[s] * randn(rng, ks))
                r = y[:, s] - Xs[s] * β
            end
            @inbounds for i = 1:n_obs
                loglik[j, i] = -0.5 * log(2π * σ²) - 0.5 * r[i]^2 / σ²
            end
        end
        for i = 1:n_obs
            col = @view loglik[:, i]
            m   = maximum(col)
            total_lppd  += m + log(sum(exp.(col .- m)) / n_samples)
            total_pwaic += _var(col)
        end
    end

    # Scale the fit term by n_fit/n, mirroring score's n_fit*log(RSS/n).
    lppd_scaled = total_lppd * (n_fit / n_obs)
    return -2 * (lppd_scaled - total_pwaic)
end

"""
    gmdl_score(y, θ, Ξ, RSS, n, k; n_eff=nothing)

gMDL (Hansen & Yu 2003, eq. 15-16), regression, unknown variance, scored on the
sparse Ξ with the caller's floored RSS.

    S = RSS/(n - k),   F = (y'y - RSS)/(k S)
    gMDL = (n_fit/2) log S + (k/2) log F + log(n_pen)     if F > 1
         = (n_fit/2) log(y'y/n) + (1/2) log(n_pen)        otherwise

Paper p.153: the code cost of the hyperparameters is log n in the upper branch
and (1/2) log n in the lower — the previous implementation used 0.5*log n in
both. (That constant is independent of k, so it cancels between two F>1 models
and changed no rankings in testing; corrected here for fidelity.)

Lower is better. `RSS` and `k` are the caller's (do NOT recompute them).
"""
function gmdl_score(y, θ, Ξ, RSS, n, k; n_eff=nothing, fit_uses_neff::Bool=true)
    if !any(abs.(Ξ) .> 0)
        return Inf
    end

    n_obs = size(y, 1)
    n_pen = isnothing(n_eff) ? float(n_obs) : float(n_eff)
    n_fit = (isnothing(n_eff) || !fit_uses_neff) ? float(n_obs) : n_pen
    kf    = float(k)                       # caller's count(nonzeros)+1

    if kf >= n_pen || (n_pen - kf) <= 0
        return Inf
    end

    R   = max(RSS, eps())                  # caller's floored, sparse-Ξ RSS
    yty = sum(abs2, y)
    FSS = yty - R

    if FSS <= 0
        return (n_fit / 2) * log(yty / n_obs) + 0.5 * log(n_pen)
    end

    S = R / (n_pen - kf)
    F = FSS / (kf * S)

    if F > 1.0
        return (n_fit / 2) * log(S) + (kf / 2) * log(F) + log(n_pen)
    else
        return (n_fit / 2) * log(yty / n_obs) + 0.5 * log(n_pen)
    end
end
function _rand_gamma(rng, shape::Real, scale::Real)
    if shape < 1
        # Boosting: Gamma(shape) = Gamma(shape+1) * U^(1/shape)
        u = rand(rng)
        return _rand_gamma(rng, shape + 1, scale) * u^(1 / shape)
    end
    d = shape - 1 / 3
    c = 1 / sqrt(9d)
    while true
        x = randn(rng)
        v = (1 + c * x)^3
        if v <= 0
            continue
        end
        u = rand(rng)
        if log(u) < 0.5 * x^2 + d - d * v + d * log(v)
            return d * v * scale
        end
    end
end

function _var(v::AbstractVector)
    m = sum(v) / length(v)
    s = 0.0
    @inbounds for x in v
        s += (x - m)^2
    end
    return s / (length(v) - 1)
end
