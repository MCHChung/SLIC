using DifferentialEquations

# Simulates ODE 
# Lib: library used to compute deriv: dudt = Lib(u)*Ξ 
# Ξ: model 
# u0: initial cond 
# ts: time array to save data 
# algo: solver for time stepping 
function SimODE(Lib::Function, Ξ::Matrix, u0::Vector, ts::Vector; algo=Tsit5())
    prob = ODEProblem((u,p,t) -> vec(Lib(u)*p), u0, (ts[1], ts[end]), Ξ)
    soln = solve(prob, algo , saveat=ts)
    return Array(soln)
end

# Alternative def for above 
# tspan: timespan , as Tuple 
# dt: timestep 
function SimODE(Lib::Function, Ξ::Matrix, u0::Vector, tspan::Tuple, dt::Float64; algo=Tsit5())
    prob = ODEProblem((u,p,t) -> vec(Lib(u)*p), u0, (ts[1], ts[end]), Ξ)
    soln = solve(prob, algo , saveat=ts)
    return Array(soln), soln.t
end