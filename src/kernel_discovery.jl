using Optimization , Optim, OptimizationOptimJL

include(srcdir("sparse_regress.jl"))

# INPUTS: 
# NPLib => the kernel component library 
# inds => matrix of cluster indices (i.e. i = 1, j=2 etc ), shape = (2, # data points) 
# kest => vector of estimated kernel coefficients (i.e. K_ij)
# ic => information criterion to score models with 

# OUTPUTS: 
# optsoln => final df value 
# Ξes => model coefficients 
# ips => inclusion probabilities for model 

function KernDDMD(NPLib::Function, inds, kest, ic::String;
        c = 0., 
        tol = 0.7,
        num_batches1 = 10 ,
        num_batches2 = 1000 ,
        maxiters = 500,
        trainpct=80
    )

    # define library as function of unknown parameter (df)
    PE_NP_Lib(a::Float64) = NPLib(inds, df=a) 
    PE_NP_Lib(a::Vector{Float64}) = NPLib(inds, df=a[1]) 

    # define optimization loss
    qt = reshape(kest[:], (length(kest[:]),1))
    function PE_NP_Loss(α, p)
        θ = PE_NP_Lib(α)
        Ξ, _   = EnAdSR(θ, qt, ic, c=c, tol=tol, num_batches=num_batches1, trainpct=trainpct)

        return sum(abs2, qt - θ*Ξ)
    end

    # define optimization problem
    lb, ub = 1.5 , 3. # lower and upper bounds for df
    α0 = [rand(lb:0.5:ub)] # random initial value 
    optf = OptimizationFunction(PE_NP_Loss, Optimization.AutoFiniteDiff())
    optprob = OptimizationProblem(optf, α0 , lb=[lb], ub=[ub])
    optsoln = solve(optprob, Optim.SAMIN(), maxiters=maxiters)

    # final round of DDMD
    θkern = NPLib(inds, df=optsoln[1]) 
    Ξes, _,  ips = EnAdSR(θkern, qt, ic, num_batches=num_batches2, tol=tol, c=c, trainpct=trainpct)

    return optsoln[1] , Ξes , ips
end