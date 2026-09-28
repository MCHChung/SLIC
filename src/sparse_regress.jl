using LinearAlgebra , StatsBase

# ============================================================================
# Sparse regression and model scoring. Includes:
#   - n_eff support in penalty terms
#   - fit_uses_neff flag selecting Position 1 (coherent n_eff everywhere) vs
#     Position 2 (raw n in fit, n_eff in penalty). Default Position 1.
#   - WAIC / NML hooks (implementations live in waic_gmdl.jl; the names
#     "waic"/"nml" here route to whatever is included).
#
# SLIC's selection is invariant to n vs n_eff in both fit and penalty (its
# leading coefficient is a positive multiplier that cancels in argmin, and the
# inside-log term shifts by a common additive constant).
# ============================================================================

function AdSR(θ, y, ic::String; iter=10, c=0., trainpct=80, abstol=1e-7, reltol=1e-7,
              n_eff=nothing, fit_uses_neff::Bool=true)

    nobs, n_state = size(y)
    bag_size = Int(floor(trainpct*nobs/100))
    traininds = sort(sample(1:nobs, bag_size, replace=false))
    testinds = [i for i=1:nobs if i ∉ traininds]
    θ_train = θ[traininds, :] ; θ_test = θ[testinds, :]
    y_train = y[traininds, :] ; y_test = y[testinds, :]

    η = c*cond(θ_train)

    # n_eff scales to the test split (scoring happens on the held-out rows)
    n_eff_test = isnothing(n_eff) ? nothing : n_eff * (100-trainpct) / 100

    min_score = Inf
    prev_smallinds = [1]

    Ξes = θ_train \ y_train
    X_prev = θ_train * Ξes

    for i=1:iter
        nzv = Ξes[Ξes .!= 0]
        isempty(nzv) && break
        λmin = minimum(abs.(nzv))
        λmax = maximum(abs.(nzv))
        λs = range(λmin, λmax, abs(1000*Int(ceil(log10(λmax/λmin + 1e-12)))) + 2)

        for λ in λs
            temp_Ξes = copy(Ξes)
            smallinds = (abs.(temp_Ξes).<λ)

            if smallinds == prev_smallinds
                continue
            end

            temp_Ξes[smallinds].=0
            any(all(abs.(temp_Ξes) .== 0, dims=1)) ? break : nothing
            for ind=1:n_state
                biginds = .!smallinds[:,ind]
                temp_Ξes[biginds,ind] = θ_train[:,biginds]\y_train[:,ind]
            end

            prev_smallinds = smallinds

            score_iter = score(y_test, θ_test, temp_Ξes, ic, η;
                               n_eff=n_eff_test, fit_uses_neff=fit_uses_neff)
            if score_iter < min_score
                Ξes = copy(temp_Ξes)
                min_score = score_iter
            end
        end

        X = θ_train * Ξes
        if _is_converged(X, X_prev, abstol, reltol)
            break
        end
        X_prev = X
    end

    return Ξes, min_score
end

function _is_converged(X, X_prev, abstol, reltol)::Bool
    Δ = norm(X .- X_prev)
    Δ < abstol && return true
    δ = Δ / norm(X)
    δ < reltol && return true
    return false
end

function EnAdSR(θ, y, ic::String;
        c = 0.,
        trainpct = 80,
        num_batches = 10,
        iter=10,
        tol = 0.7,
        n_eff = nothing,
        fit_uses_neff::Bool = true,
    )
    ΞB = zeros((size(θ,2), size(y,2), num_batches))
    scores_ = zeros(num_batches)
    N = size(y,1)

    for i=1:num_batches
        Ξes, sc = AdSR(θ, y, ic; iter=iter, c=c, trainpct=trainpct,
                       n_eff=n_eff, fit_uses_neff=fit_uses_neff)
        ΞB[:,:,i] = Ξes
        scores_[i] = sc
    end

    biginds = abs.(ΞB).>0
    ips = mean(biginds, dims=3)

    Ξes = sum(ΞB, dims=3)./max.(count(biginds, dims=3), 1)
    Ξes[ips .< tol] .= 0
    Ξes = Ξes[:,:,1]

    n_state = size(y, 2)
    smallinds = .!(abs.(Ξes) .> 0)
    for ind=1:n_state
        biginds = .!smallinds[:,ind]
        any(biginds) && (Ξes[biginds,ind] = θ[:,biginds]\y[:,ind])
    end

    return Ξes, scores_, ips[:,:,1]
end

# ============================================================================
# score() — Position-1 (coherent n_eff) by default. See header of
# patch3/score_position1.jl for the full rationale.
# ============================================================================

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
function score(y, θ, Ξ, ic, η; n_eff=nothing, fit_uses_neff::Bool=true)
    n = size(y, 1)
    n_pen = isnothing(n_eff) ? n : n_eff
    n_fit = (isnothing(n_eff) || !fit_uses_neff) ? n : n_pen

    k = count(abs.(Ξ) .> 0.) + 1
    RSS = sum(abs2, y - θ*Ξ) + η

    if ic == "slic"
        return n*log(k*RSS/n)

    elseif ic == "aic"
        return n_fit*log(RSS/n) + 2*k

    elseif ic == "aicc"
        return n_fit*log(RSS/n) + 2*k*n_pen/(n_pen-k-1)

    elseif ic == "bic"
        return n_fit*log(RSS/n) + k*log(n_pen)

    elseif ic == "hqic"
        return n_fit*log(RSS/n) + k*log(log(n_pen))

    elseif ic == "bc"
        return n_fit*log(RSS/n) + n_pen^(1/3) * sum(1/i for i=1:k)

    elseif ic == "Cp"
        return RSS*(1+2*k/n_pen)

    elseif ic == "kic"
        kic = 0
        biginds = abs.(Ξ) .> 0
        if size(y,2) > 1
            for ind=1:size(y,2)
                biginds_i = biginds[:,ind]
                kic += log(abs(det((RSS/n)^-1 * θ[:,biginds_i]'*θ[:,biginds_i])))
            end
        else
            kic += log(abs(det((RSS/n)^-1 * θ[:,vec(biginds)]'*θ[:,vec(biginds)])))
        end
        kic += n_fit*log(RSS/n) - k*log(2*π)
        return kic

    elseif ic == "waic"
        return waic_score(y, θ, Ξ, RSS, n; n_eff=n_pen)

    elseif ic == "nml"
        return nml_score(y, θ, Ξ, RSS, n, k; n_eff=n_pen)

    else
        throw("Invalid IC. Valid: slic, aic, aicc, bic, hqic, bc, Cp, kic, waic, nml")
    end
end

# WAIC/NML implementations are provided by a separate file when those criteria
# are needed (waic_gmdl.jl). The n_eff benchmarks do NOT use the waic/nml
# branches, so we do not include it here by default — the classical
# criteria and SLIC are fully self-contained above. Scripts that need
# waic/gmdl include their implementation explicitly (see waic_gmdl_benchmarks.jl).
