using LinearAlgebra, Optim, FFTW 

# This is a Julia implementation of a simple Whittaker-Henderson filter found in: 
# https://www.sciencedirect.com/science/article/abs/pii/S0167947309003491

# INPUTS: 
# y => noisy data 

# OUTPUTS
# ŷ => smoothed data 

function smooth_ode(y::AbstractVector; tol=1e-5)
    
    # cross-val function for auto-determining smoothing hyperparameter
    function GCVscore(p::Float64, DCTy, Lambda, n)
        s = 10^p
        Gamma = 1 ./ (1 .+ s .* Lambda.^2)
        RSS = norm(DCTy[:] .* (Gamma[:] .-1))^2
        TrH = sum(Gamma[:])
        GCVs = RSS/n/(1 .- TrH/n)^2
        return GCVs
    end
    
    y = reshape(y, (length(y),1))
    n1, n2 = size(y)
    n = n1 * n2
    Lambda = -2 .+ 2 .* cos.((0:n2-1)' .* π ./ n2) .+ (-2 .+ 2 .* cos.((0:n1-1) .* π ./ n1)) 

    goodinds = isfinite.(y) # find bad or missing data 
    W = ones((n1,n2)).*goodinds # set 'bad' data to have zero weights 
    #initial guesses 
    ym = y .* goodinds 
    ŷ = zeros(size(y))
    err = Inf
    while err > tol
        DCTy = dct(W.*(ym-ŷ) + ŷ)
        res = optimize(p -> GCVscore(p, DCTy, Lambda, n), -15, 38)
        Gamma = 1 ./ (1 .+ 10.0^res.minimizer .* Lambda.^2)
        ŷnew = idct(Gamma .* DCTy)
        err = norm(ŷ .- ŷnew)/norm(ŷ)
        ŷ = ŷnew
    end

    return ŷ
end

# does the same as the above with multiple state variables 
function smooth_ode(un::AbstractMatrix) 
    u = similar(un)
    for row=1:size(un,1)
        u[row, :] = smooth_ode(un[row,:])
    end
    return u
end

# this does the same as the above, but with higher-dim data 
function smooth_pde(y; tol = 1e-5) 
    # y => Noisy data
    # W => Weights for datapoints. Typically set to W = ones(size(y))

    # gets cross-val score
    function GCVscore(p::Float64, DCTy, Lambda, n)
        s = 10^p
        Gamma = 1 ./ (1 .+ s .* Lambda.^2)
        RSS = norm(DCTy[:] .* (Gamma[:] .-1))^2
        TrH = sum(Gamma[:])
        GCVs = RSS/n/(1 .- TrH/n)^2
        return GCVs
    end

    ns = size(y)
    n = prod(ns)
    Lambda = zeros(ns)
    for i=1:length(ns)
        l = @. -2 + 2 * cos((0:ns[i]-1) * π / ns[i])
        # reshape l 
        s = replace(x -> x!=ns[i] ? 1 : ns[i], ns)
        l = reshape(l , s)
        # update Lambda
        Lambda .+= l
    end

    goodinds = isfinite.(y) # find bad or missing data 
    W = ones(ns).*goodinds # set 'bad' data to have zero weights 
    #initial guesses 
    ym = y .* goodinds 
    ŷ = zeros(ns)
    err = Inf
    while err > tol
        DCTy = dct(W.*goodinds.*(ym-ŷ) + ŷ)
        res = optimize(p -> GCVscore(p, DCTy, Lambda, n), -15, 38)
        Gamma = 1 ./ (1 .+ 10.0^res.minimizer .* Lambda.^2)
        ŷnew = idct(Gamma .* DCTy)
        err = norm(ŷ .- ŷnew)/norm(ŷ)
        ŷ = ŷnew
    end

    return ŷ
end