using LinearAlgebra

using LinearAlgebra

"""
    SparseReg(Theta, char_sizes, valid_single, opts)

Sparse regression enforcing 0 = Theta * Xi.  
Faithful MATLAB → Julia conversion.

Returns:
    Xi, lambda, best_term, lambda1
"""
function Sp_SparseReg(Theta::AbstractMatrix;
                   char_sizes=nothing,
                   valid_single=nothing,
                   opts=Dict())

    # === Read options ===
    threshold     = get(opts, :threshold, "multiplicative")
    brute_force   = get(opts, :brute_force, 1)
    delta         = get(opts, :delta, 1e-15)
    gamma         = get(opts, :gamma, 1.25)
    epsilon       = get(opts, :epsilon, 1e-2)
    verbose       = get(opts, :verbose, 0)
    n_terms       = get(opts, :n_terms, -1)

    h, w = size(Theta)

    # === Renormalize by characteristic sizes (MATLAB: Theta(:,term) /= char_sizes(term)) ===
    if char_sizes !== nothing
        for term in 1:w
            Theta[:, term] /= char_sizes[term]
        end
    end

    # If valid_single was not provided:
    if valid_single === nothing
        valid_single = ones(w)
    end

    # === Initial SVD ===
    U, S, V = svd(Theta)
    Xi = V[:, end]

    if verbose > 0
        Sigmas = diag(S)
        Sigmas = Sigmas[Sigmas .> 0]
        @show V
        @show log.(Sigmas ./ minimum(Sigmas))
    end

    lambda = norm(Theta * Xi)

    if verbose > 0
        @show lambda
    end

    # === Best one-term model ===
    nrm = zeros(w)
    for term in 1:w
        nrm[term] = norm(Theta[:, term]) / valid_single[term]
        if verbose > 0
            @show nrm[term]
        end
    end

    lambda1, ind_single = findmin(nrm)

    # === Sparse pruning structures ===
    smallinds = falses(w)
    margins   = zeros(w)
    lambdas   = zeros(w+1)
    lambdas[1] = lambda

    # Xis holds all intermediate Xi (like MATLAB cell array)
    Xis = Vector{Vector{Float64}}()
    if threshold != "multiplicative"
        push!(Xis, copy(Xi))
    end

    product = zeros(w)
    res_inc = fill(Inf, w)

    # ========================================================================
    #                           MAIN ITERATION
    # ========================================================================
    for i in 1:min(100, w)

        if threshold != "multiplicative"
            push!(Xis, copy(Xi))
        end

        if brute_force == 1
            res_inc .= Inf
        end

        # --------------------------------------------------------------------
        # Try removing each possible column
        # --------------------------------------------------------------------
        for p_ind in 1:w
            if brute_force == 1
                if !smallinds[p_ind]
                    # Try dropping p_ind
                    small_copy = copy(smallinds)
                    small_copy[p_ind] = true

                    Xi_copy = copy(Xi)
                    Xi_copy[p_ind] = 0

                    _, _, Vtmp = svd(Theta[:, .!small_copy])
                    Xi_copy[.!small_copy] = Vtmp[:, end]

                    res_inc[p_ind] = norm(Theta * Xi_copy) / lambda
                end
            else
                # Non–brute force projection (as in MATLAB)
                col = copy(Theta[:, p_ind])
                for q_ind in 1:w
                    if p_ind != q_ind && !smallinds[q_ind]
                        other = Theta[:, q_ind]
                        col -= (dot(col, other) / norm(other)^2) * other
                    end
                end

                # row norms: sqrt(sum(Theta.^2,2))
                rownorms = sqrt.(sum(abs2, Theta; dims=2))
                product[p_ind] = norm(Xi[p_ind] * col ./ rownorms)
            end
        end

        # --------------------------------------------------------------------
        # Choose the column to eliminate
        # --------------------------------------------------------------------
        if brute_force == 1
            Y, I = findmin(res_inc)
            margins[i] = Y

            if verbose > 0
                @show res_inc
                if threshold != "multiplicative"
                    @show i lambda
                end
            end

            if (Y <= gamma) || (threshold != "multiplicative")
                smallinds[I] = true
                Xi[I] = 0

                _, _, Vtmp = svd(Theta[:, .!smallinds])
                Xi[.!smallinds] = Vtmp[:, end]

                lambda = norm(Theta * Xi)
                lambdas[i+1] = lambda

                if count(!, smallinds) == 1
                    break
                end
            else
                if verbose > 0
                    @show Y I
                end
                break
            end

        else
            # Non-brute-force criteria
            product[smallinds] .= Inf
            Y, I = findmin(product)
            smallinds[I] = true

            if count(!, smallinds) == 0
                break
            end

            if verbose > 0
                @show product'
            end

            Xi_old = copy(Xi)
            Xi[smallinds] .= 0.0

            _, _, Vtmp = svd(Theta[:, .!smallinds])
            Xi[.!smallinds] = Vtmp[:, end]

            lambda_old = lambda
            lambda = norm(Theta * Xi)
            lambdas[i+1] = lambda

            margin = lambda / lambda_old
            margins[i] = margin

            if verbose > 0
                @show lambda margin
            end

            if (margin > gamma) && (lambda > delta) && (threshold == "multiplicative")
                Xi = Xi_old
                break
            end
        end
    end

    # Append final Xi
    push!(Xis, copy(Xi))

    # ========================================================================
    #                       THRESHOLD POST-PROCESSING
    # ========================================================================
    if threshold == "pareto"
        Ymar, Imar = findmax(margins)
        if n_terms > 1
            Imar = length(margins) - n_terms + 1
        end
        Xi = copy(Xis[Imar])
        lambda = norm(Theta * Xi)

    elseif threshold == "error"
        # find first index where lambdas > epsilon * lambda1
        idxs = findall(x -> x > epsilon*lambda1, lambdas)
        if isempty(idxs)
            Ism = 1
        else
            Ism = max(first(idxs) - 1, 1)
        end
        Xi = copy(Xis[Ism])
        lambda = norm(Theta * Xi)
    end

    # ========================================================================
    # Compare to single-term model, renormalize if needed
    # ========================================================================
    best_term = ind_single

    if char_sizes !== nothing
        Xi = Xi ./ char_sizes
    end

    Xi_sc = Xi[2:end]/Xi[1] 
    Xi_sc = replace(Xi_sc, -0. => 0.) 
    return Xi, Xi_sc, lambda, best_term, lambda1
end

function Sp_SparseReg_2(θ, qt; 
    char_sizes=nothing,
    valid_single=nothing,
    opts=Dict()
)   
    Ξ = zeros(size(θ,2)+1, size(qt,1))
    Ξsc = zeros(size(θ,2), size(qt,1))
    for i=1:size(qt,1)
        θsp = hcat(qt[i,:], -θ)
        Ξi, Ξsc_i, _ =  Sp_SparseReg(θsp, char_sizes=char_sizes, valid_single=valid_single, opts=opts)
        Ξ[:,i] = Ξi
        Ξsc[:,i] = Ξsc_i
    end
    return Ξ, Ξsc
end

