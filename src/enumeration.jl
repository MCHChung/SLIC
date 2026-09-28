using LinearAlgebra, Combinatorics

# ============================================================================
# Exhaustive enumeration over all sparsity patterns.
#
# For libraries small enough that 2^p is tractable (VdP: 2^9=512,
# Pendulum: 2^10=1024, Lorenz per-eq: 2^19≈5e5), enumerate every support,
# fit OLS, score with each criterion, report which criterion picks
# the best model.
#
# This isolates the criterion from the candidate-generation procedure
# (Algorithm 1's threshold sweep), removing any confound from that loop.
# ============================================================================

"""
    enumerate_supports(θ, y; ics=["slic","aic",...], n_eff=nothing, max_k=nothing)

Enumerate all subsets of columns of `θ`, fit OLS for each, score with each
criterion in `ics`, and return the best-scoring support per criterion.

Inputs:
  θ : n × p library matrix (with bias term as one column if applicable)
  y : n × n_state response (will iterate per-state for multidim targets)
  ics : vector of criterion names (strings; see score() in sparse_regress.jl)
  n_eff : optional effective sample size for competitor penalties
  max_k : optional cap on support size (default: p, unlimited)

Output: Dict with keys = criterion names, values = best-scoring Ξ matrix.
Also returns Dict of (score, support) tuples for diagnostics.
"""
function enumerate_supports(θ::AbstractMatrix, y::AbstractMatrix, ics::Vector{String};
                            n_eff=nothing, max_k=nothing, η=0.0, verbose=false)
    n, p = size(θ)
    n_state = size(y, 2)
    max_k = isnothing(max_k) ? p : min(max_k, p)

    best_score = Dict(ic => Inf for ic in ics)
    best_Ξ    = Dict(ic => zeros(p, n_state) for ic in ics)
    best_support = Dict(ic => Int[] for ic in ics)

    n_supports = 0

    # Loop over support sizes from 1 to max_k
    for k_supp = 1:max_k
        for inds in combinations(1:p, k_supp)
            n_supports += 1
            if verbose && n_supports % 10000 == 0
                println("  enumerated $n_supports supports so far ...")
                flush(stdout)
            end

            # Fit OLS on this support
            θ_sub = θ[:, inds]
            # Skip if rank-deficient
            if rank(θ_sub) < length(inds)
                continue
            end

            Ξ_sub = θ_sub \ y                     # k_supp × n_state
            Ξ_full = zeros(p, n_state)
            Ξ_full[inds, :] = Ξ_sub

            # Score with each criterion
            for ic in ics
                sc = score(y, θ, Ξ_full, ic, η; n_eff=n_eff)
                if sc < best_score[ic]
                    best_score[ic] = sc
                    best_Ξ[ic] = Ξ_full
                    best_support[ic] = collect(inds)
                end
            end
        end
    end

    if verbose
        println("  enumeration complete: $n_supports supports evaluated")
    end

    return best_Ξ, best_score, best_support
end

"""
    enumerate_per_equation(θ, y; ics, ...)

For multi-output regression where each equation is fit separately
(typical for SINDy: one Ξ column per state variable), enumerate
supports independently per equation. Returns Ξ matrices stitched
back together.

This matches what Algorithm 1 effectively does and dramatically reduces
search space: instead of 2^p combined supports, enumeration is 2^p per
equation. For Lorenz this is 2^19 per equation × 3 equations vs
2^57 joint.
"""
function enumerate_per_equation(θ::AbstractMatrix, y::AbstractMatrix, ics::Vector{String};
                                 n_eff=nothing, max_k=nothing, η=0.0, verbose=false)
    n, p = size(θ)
    n_state = size(y, 2)
    max_k = isnothing(max_k) ? p : min(max_k, p)

    Ξ_combined = Dict(ic => zeros(p, n_state) for ic in ics)
    sc_per_state = Dict(ic => zeros(n_state) for ic in ics)

    for s = 1:n_state
        y_s = reshape(y[:, s], (n, 1))
        best_Ξ_s, best_sc_s, _ = enumerate_supports(θ, y_s, ics;
                                                     n_eff=n_eff, max_k=max_k,
                                                     η=η, verbose=verbose)
        for ic in ics
            Ξ_combined[ic][:, s] = best_Ξ_s[ic][:, 1]
            sc_per_state[ic][s] = best_sc_s[ic]
        end
    end

    return Ξ_combined, sc_per_state
end
