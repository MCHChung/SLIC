using Catalyst, DiffEqParamEstim, OptimizationOptimJL, DifferentialEquations

# function serves to build odes to estimate the kernel of the RNA-NP system. It: 
# 1. Uses Catalyst.jl to build the equations with a kernel we want to estimate 
# 2. Uses DiffEqParamEstim.jl to determine the kernel values that minimize the L2 loss with data 
# 3. Returns the estimated kernel, the indices used, and other quantities 

#=
=========== INPUTS ==============
N: number of particles to simulate 
u0vec: initial particle count 
kest: estimated kernel values

=========== OUTPUTS ==============
tsteps: time points
soln: simulated data
=#

function SimAgg(N::Int , u0vec::AbstractVector, kest::AbstractVector; 
        odealgo = Tsit5() , # solver for ode system
        tspan = (0.0f0, 27.0f0) 
    )
    
    # ========================= GENERATE ODE =====================================
   
    integ(x) = Int(floor(x))
    n        = integ(N/2)
    nr       = N%2 == 0 ? (n*(n + 1) - n) : (n*(n + 1)) # No. of forward reactions

    # POSSIBLE PAIRS OF REACTING MULTIMERS
    pair = []
    for i = 2:N
        push!(pair,[1:integ(i/2)  i .- (1:integ(i/2))])
    end
    pair = vcat(pair...) ;
    vᵢ = @view pair[:,1] ;  # Reactant 1 indices
    vⱼ = @view pair[:,2] ; # Reactant 2 indices  
    sum_vᵢvⱼ = @. vᵢ + vⱼ ; # Product index


    # USE ModelingToolkit TO BUILD PARAMS & VARS
    # state variables are X, pars stores rate parameters for each rx
    @parameters t k[1:nr] # ks had to be added to parameters
    @species X(t)[1:N] # defining the reactions and species 
    pars = Pair.(collect(k), kest)

    u₀map = Pair.(collect(X), u0vec)   # map variable to its initial value
    tsteps = range(tspan[1], tspan[2], length = 28)

    # BUILD Catalyst REACTION SYSTEM 
    # vector to store the Reactions in
    rx = []
    for n = 1:nr
        # for clusters of the same size, double the rate
        if (vᵢ[n] == vⱼ[n])
            # form of the Reaction arg ==> Reaction(rate, [species in], [species out], [stoic coef in], [stoic coef out])
            push!(rx, Reaction(k[n], [X[vᵢ[n]]], [X[sum_vᵢvⱼ[n]]], [2], [1])) 
        else
            push!(rx, Reaction(k[n], [X[vᵢ[n]], X[vⱼ[n]]], [X[sum_vᵢvⱼ[n]]],
                            [1, 1], [1]))
        end
    end

    @named rs = ReactionSystem(rx, t, collect(X), collect(k))

    # BUILD AND SOLVE ODE PROBLEM
    # trying conversion to an ODE system as well
    odesys = convert(ODESystem, rs)
    odeprob = ODEProblem(odesys,  u₀map, tspan, pars)
    soln = Array(solve(odeprob, odealgo, saveat=tsteps))

    return tsteps, soln #  time points, indices of kernel, initial guess, kernel estimation from opt prob, simulated data
end