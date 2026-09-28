# ============================================================================
# OVERRIDE for the score_on_all probe. Redefines ONLY AdSR and EnAdSR.
#
# include AFTER common.jl; Julia takes the last definition, so this shadows
# the versions from src/sparse_regress.jl WITHOUT modifying that file. Method
# overwrite warnings on load are expected and harmless.
#
# `score` and `_is_converged` are unchanged and are NOT redefined here.
#
# What changes: AdSR gains `score_on_all` (default false). When false the
# behaviour is byte-identical to src/sparse_regress.jl — fit on train, score on
# the 20% holdout, n_eff scaled to that split. When true, the fit still happens
# on the train split (so EnAdSR's batches still vary and tol=0.7 still
# aggregates something real) but scoring moves to ALL n with the unscaled n_eff.
#
# Why: an IC's penalty corrects the in-sample optimism of a fit AT THE SAMPLE
# SIZE IT IS SCORED AT. Scoring at 0.2n while the penalty assumes n applies the
# wrong n. BIC's per-parameter acceptance threshold is exp(log(n)/n)-1: at
# Lorenz's n_eff_test = 0.2*442 = 88 that is 5.20%, but at the full n_eff = 442
# it is 1.39% — a uniform 3.7x handicap across every system that makes the
# classical criteria artificially conservative. SLIC is unaffected either way,
# since n cancels in the comparison of n*log(k*RSS/n).
# ============================================================================

function AdSR(θ, y, ic::String; iter=10, c=0., trainpct=80, abstol=1e-7, reltol=1e-7,
              n_eff=nothing, fit_uses_neff::Bool=true, score_on_all::Bool=false,
              scale_eta::Bool=true)

    nobs, n_state = size(y)
    bag_size = Int(floor(trainpct*nobs/100))
    traininds = sort(sample(1:nobs, bag_size, replace=false))
    testinds = [i for i=1:nobs if i ∉ traininds]
    θ_train = θ[traininds, :] ; θ_test = θ[testinds, :]
    y_train = y[traininds, :] ; y_test = y[testinds, :]

    η = c*cond(θ_train)

    # Scoring target. Default (score_on_all=false) is the standard pipeline
    # behaviour: fit on train, score on the held-out rows, with n_eff scaled to
    # that split.
    #
    # score_on_all=true: fit on train (so the ensemble still varies across
    # batches), but score on ALL n. Rationale: an information criterion's
    # penalty is derived to correct the in-sample optimism of a fit at the
    # sample size it is scored at. Scoring at 0.2n while the penalty assumes n
    # applies the wrong n: BIC's per-parameter acceptance threshold is
    # exp(log(n)/n)-1, which at n_eff_test = 0.2*442 = 88 is 5.20% but at the
    # full n_eff = 442 is 1.39% — a 3.7x handicap that makes the classical
    # criteria artificially conservative. SLIC is unaffected either way, since
    # n cancels in the comparison of n*log(k*RSS/n).
    θ_score = score_on_all ? θ : θ_test
    y_score = score_on_all ? y : y_test
    n_eff_score = if isnothing(n_eff)
        nothing
    elseif score_on_all
        n_eff
    else
        n_eff * (100-trainpct) / 100
    end

    # CONFOUND GUARD. score() forms RSS = sum(abs2, y_score - θ_score*Ξ) + η.
    # RSS is extensive in the number of scored rows, but η = c*cond(θ_train) is
    # not: scoring on all n makes RSS ~5x larger while η is unchanged, so the
    # relative floor η/RSS silently weakens ~5x. That is a SECOND change riding
    # along with the intended n change, and both push toward over-selection —
    # they cannot be separated. Rescale η to the scoring set so that η/RSS is
    # preserved and the n change is isolated.
    #   scale_eta=true  (default): isolates the n effect  <- what we want to measure
    #   scale_eta=false          : floor also weakens; the two effects are mixed
    η_score = if score_on_all && scale_eta && !isempty(testinds)
        η * (nobs / length(testinds))
    else
        η
    end

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

            score_iter = score(y_score, θ_score, temp_Ξes, ic, η_score;
                               n_eff=n_eff_score, fit_uses_neff=fit_uses_neff)
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

function EnAdSR(θ, y, ic::String;
        c = 0.,
        trainpct = 80,
        num_batches = 10,
        iter=10,
        tol = 0.7,
        n_eff = nothing,
        fit_uses_neff::Bool = true,
        score_on_all::Bool = false,
        scale_eta::Bool = true,
    )
    ΞB = zeros((size(θ,2), size(y,2), num_batches))
    scores_ = zeros(num_batches)
    N = size(y,1)

    for i=1:num_batches
        Ξes, sc = AdSR(θ, y, ic; iter=iter, c=c, trainpct=trainpct,
                       n_eff=n_eff, fit_uses_neff=fit_uses_neff,
                       score_on_all=score_on_all, scale_eta=scale_eta)
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
