using LinearAlgebra , StatsBase 

# INPUTS (for AdSR)
# Θ: The library matrix with size n x p, where n is the data length, p is the number of nonlinear basis. 
# y: Output, for dynamics will be estimated or measured derivative of dynamics.
# iter: Number of regressions you would like to perform.
# c : adds optional additional sparsity enforcement, essentially like injecting small amount of noise to system.  

# OUTPUTS (for AdSR)
# Ξes = estimated model 
# min_score = min IC score

function AdSR(θ, y, ic::String; iter = 10, c=0., trainpct=80, abstol=1e-7, reltol = 1e-7)
    @assert ic == "slic" || ic == "aicc" || ic == "bic"

    # random train-val split 
    nobs , n_state = size(y)
    bag_size = Int(floor(trainpct*nobs/100))
    traininds = sort(sample(1:nobs, bag_size, replace=false))
    testinds = [i for i=1:nobs if i ∉ traininds]
    θ_train = θ[traininds, :] ; θ_test = θ[testinds, :]
    y_train = y[traininds, :] ; y_test = y[testinds, :]

    # define information criterion scoring procedure 
    η = c*cond(θ_train) # this is optional enforcement to prevent log(RSS) -> ∞ , SLIC does well without it!
    function score(Ξ, ic)
        nobs_test = size(y_test,1) # number of observations in test/val 
        k = count(abs.(Ξ) .> 0.) + 1 # number of free params
        RSS = sum(abs2, y_test - θ_test*Ξ)
        if ic=="slic"
            nobs_test*log(k*(RSS + η)/nobs_test)
        elseif ic=="aicc"
            nobs_test*log((RSS + η)/nobs_test) + 2*k*nobs_test/(nobs_test-k-1)
        elseif ic=="bic"
            nobs_test*log((RSS + η)/nobs_test) + k*log(nobs_test)
        else
            throw("Please enter valid selection criterion => slic, aicc, bic")
        end
    end
    # Initialize comparison values
    min_score = Inf
    prev_smallinds = [1]

    # Get an initial estimate of the selection matrix Ξes
    Ξes = θ_train \ y_train    
    X_prev = θ_train * Ξes 

    for i=1:iter
        
        # auto-gen thresholds 
        λmin = min(abs.(Ξes[Ξes .!= 0])...) 
        λmax = max(abs.(Ξes[Ξes .!= 0])...)
        λs = range(λmin, λmax, abs(1000*Int(ceil(log10(λmax/λmin)))))
        
        # sparsify and score effect 
        for λ in λs
            # Make a temporary Ξes matrix to test out the effect of λ
            temp_Ξes = copy(Ξes)
            # Get the index of values whose absolute value is smaller than λ
            smallinds = (abs.(temp_Ξes).<λ)

            # If the effect of λ is the same as the previous one, no need to do calculations again
            if smallinds == prev_smallinds
                continue
            end

            # Set the parameter value of library term whose absolute value is smaller than λ as zero
            temp_Ξes[smallinds].=0
            any(all(abs.(temp_Ξes) .== 0, dims=1)) ? break : nothing
            # Regress the dynamics to the remaining terms
            for ind=1:n_state
                biginds = .!smallinds[:,ind]
                temp_Ξes[biginds,ind] = θ_train[:,biginds]\y_train[:,ind]
            end
            
            # Save the current small indices
            prev_smallinds = smallinds

            # calculate the loss and compare it to our best loss
            score_iter = score(temp_Ξes, ic)
            if score_iter < min_score
                Ξes = copy(temp_Ξes)
                min_score = score_iter
            end
        end
        
        X = θ_train * Ξes # make new prediction
        
        # If nothing, or very little, changed in one iteration, then we have converged
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

# This does AdSR with ensembling. The final output, ips, is the matrix of inclusion probabilities
function EnAdSR(θ, y, ic::String;
        c = 0.,
        trainpct = 80,
        num_batches = 10,
        iter=10,
        tol = 0.7
    )
    # hold ensemble of models  
    ΞB = zeros((size(θ\y)..., num_batches))
    scores = zeros(num_batches)
    # determine number of sample size
    N = size(y,1)

    for i=1:num_batches
        # get model and score for this random train-val split 
        Ξes , score = AdSR(θ, y, ic; iter=iter, c=c, trainpct=trainpct)
        # add to holders 
        ΞB[:,:,i] = Ξes
        scores[i] = score
    end

    # compute the inclusion probabilities for model coefficients 
    biginds = abs.(ΞB).>0
    ips = mean(biginds , dims=3)

    # Compute the ensembled Ξs and probabilistically prune
    Ξes = sum(ΞB, dims=3)./count(biginds,dims=3) 
    Ξes[ips .< tol] .= 0 
    Ξes = Ξes[:,:,1]

    # final regression to update
    n_state = size(y, 2)
    smallinds = .!(abs.(Ξes) .> 0)
    for ind=1:n_state
        biginds = .!smallinds[:,ind]
        Ξes[biginds,ind] = θ[:,biginds]\y[:,ind]
    end

    return Ξes, scores, ips[:,:,1]
end

