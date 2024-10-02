using Catalyst, DiffEqParamEstim, OptimizationOptimJL, DifferentialEquations

# function serves to build odes to estimate the kernel of the RNA-NP system. It: 
# 1. Uses Catalyst.jl to build the equations with a kernel we want to estimate 
# 2. Uses DiffEqParamEstim.jl to determine the kernel values that minimize the L2 loss with data 
# 3. Returns the estimated kernel, the indices used, and other quantities 

#=
=========== INPUTS ==============
N: number of particles to simulate 
Xn: raw data 
u0vec: initial particle count 
k0: initial guess of kernel 

=========== OUTPUTS ==============
tsteps: time points
Matrix(pair'): indices of kernel 
k0: initial guess for kernel 
optsol: kernel estimation from opt prob 
simdata: simulated data
=#
function EstimKernel(N::Int , Xn::AbstractMatrix, u0vec::AbstractVector, k0::AbstractVector; 
        i = 4,
        Vₒ = (4π/3)*(10e-06*100)^3 , # volume of a monomers in cm³
        uₒ = 10000 , # initial monomer number    
        df = 1.5 , # fractal dimension 
        γ = 1  , # exponent for product kernel 
        C = 1.84e-04 , # cm³ s⁻¹ 
        odealgo = Tsit5() , # solver for ode system
        maxiters = 100000, # max iterations for kernel estim algorithm
        use_rand_k0 = false, # use a random k0 for kernel estimation
        lb_mag = 1e-3, # lower bound for all kernel elements
        ub_mag = 1e1, # upper bound for all kernel elements
        optalgo = BFGS(), # optimization algorithm for kernel
        tspan = (0.0f0, 27.0f0) 
    )
    
    # ========================= GENERATE ODE =====================================
    # PARAMETERS USED 
    Nₒ = 1e-06/Vₒ                # initial conc. = (No. of init. monomers) / bulk volume
    V = uₒ/Nₒ                    # Bulk volume of system in cm³

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
    volᵢ = Vₒ*vᵢ      ;    
    volⱼ = Vₒ*vⱼ     ;      
    sum_vᵢvⱼ = @. vᵢ + vⱼ ; # Product index

    # BUILD KERNEL VALUES
    # This shouldn't make much of a difference. We aren't building a specific model, just the ODEs
    if i==1
        B = 1.53e03                # s⁻¹
        kv = @. B*(volᵢ + volⱼ)/V  ; # dividing by volume as its a bi-molecular reaction chain
        #kv = @. 1e-2 * (vᵢ + vⱼ)
    elseif i==2           
        kv = fill(C/V, nr) ; # constant kernel 
    elseif i==3
        kv = @. (C/V)  * (vᵢ * vⱼ)^γ ; # product kernel 
    elseif i==4
        kv = @. (C/V) * (1 + 0.5*(vᵢ/vⱼ)^(1/df)+ 0.5*(vⱼ/vᵢ)^(1/df)) ; # Full brownian kernel 
    end

    # USE ModelingToolkit TO BUILD PARAMS & VARS
    # state variables are X, pars stores rate parameters for each rx
    @parameters t k[1:nr] # ks had to be added to parameters
    @species X(t)[1:N] # defining the reactions and species 
    pars = Pair.(collect(k), kv)

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

    # ================================ ESTIMATE KERNEL ====================================
    # build the cost function w/ L2
    cost_func =  build_loss_objective(odeprob, odealgo, L2Loss(tsteps,Xn), Optimization.AutoForwardDiff(),
    maxiters=maxiters,verbose=false)

    # Build and solve Optimization problem. Return kernel estimation
    if isempty(k0) && use_rand_k0
        k0 = rand(range(lb_mag, ub_mag, 100*Int(abs(log10(lb_mag) - log10(ub_mag)))), nr)
    elseif isempty(k0) && !use_rand_k0
        k0 = lb_mag*ones(nr)
    end

    optprob = Optimization.OptimizationProblem(cost_func, k0, lb = lb_mag*ones(length(k0)), ub = ub_mag*ones(length(k0)))
    optsol = solve(optprob, optalgo)

    # solve the ode and spit back the solution
    prob = ODEProblem(odesys,  u₀map, tspan, Pair.(collect(k), optsol))
    simdata = Array(solve(prob, odealgo, saveat=tsteps))

    return tsteps, Matrix(pair'), k0 , optsol, simdata #  time points, indices of kernel, initial guess, kernel estimation from opt prob, simulated data
end