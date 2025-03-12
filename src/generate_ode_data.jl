using DifferentialEquations 

# This generates the simulated ODE data from the main text 
# sys = system you want returned 
# u0 = initial cond 
# ps = parameters for system 
# tspan = timespan of simulation 
# dt = fixed timestep of simulation 
function GenData(sys::Int, tspan, dt, u0::Vector, ps::Vector; algo=Tsit5())
    @assert 0 < sys < 7
    if sys==1 # Lorenz
        @assert length(ps) == 3 && length(u0) == 3 
        # define derivative 
        function lor!(du, u, p, t)
            σ, ρ, β = p[1], p[2], p[3]
            du[1] = σ*(u[2] - u[1])
            du[2] = u[1]*(ρ - u[3]) - u[2]
            du[3] = u[1]*u[2] - β*u[3]
        end
        # solve problem 
        prob = ODEProblem(lor!, u0, tspan, ps)
        soln = solve(prob, algo , saveat=dt)
        return Array(soln), soln.t
    elseif sys==2 # Rossler
        @assert length(ps) == 3 && length(u0) == 3 
        # define derivative 
        function ross!(du, u, p, t)
            a, b, c = p[1], p[2], p[3]
            du[1] = -u[2]-u[3]
            du[2] = u[1] + a*u[2]
            du[3] = b + u[3]*(u[1]-c)
        end
        # solve problem 
        prob = ODEProblem(ross!, u0, tspan, ps)
        soln = solve(prob, algo , saveat=dt)
        return Array(soln), soln.t
    elseif sys==3 # LV
        @assert length(ps) == 1 && length(u0) == 2 
        # define derivative 
        function lv!(du, u, p, t)
            r = p[1]
            du[1] = u[1] - r*u[1]*u[2] 
            du[2] = r*u[1]*u[2] - u[2]
        end
        # solve problem 
        prob = ODEProblem(lv!, u0, tspan, ps)
        soln = solve(prob, algo , saveat=dt)
        return Array(soln), soln.t
    elseif sys==4 # Brus
        @assert length(ps) == 2 && length(u0) == 2 
        # define derivative 
        function brus!(du, u, p, t)
            a,b = p[1], p[2]
            du[1] = 1. + a*u[1] + b*u[1]^2*u[2]
            du[2] = (-a-1)*u[1] - b*u[1]^2*u[2] 
        end
        # solve problem 
        prob = ODEProblem(brus!, u0, tspan, ps)
        soln = solve(prob, algo , saveat=dt)
        return Array(soln), soln.t
    elseif sys==5 # VdP
        @assert length(ps) == 1 && length(u0) == 2 
        # define derivative 
        function vdp!(du, u, p, t)
            μ = p[1]
            du[1] = u[2]
            du[2] = μ*(1-u[1]^2)*u[2] - u[1]
        end
        # solve problem 
        prob = ODEProblem(vdp!, u0, tspan, ps)
        soln = solve(prob, algo , saveat=dt)
        return Array(soln), soln.t
    elseif sys==6 # NLP
        @assert length(ps) == 1 && length(u0) == 2 
        # define derivative 
        function nlp!(du, u, p, t)
            ω = p[1]
            du[1] = u[2]
            du[2] = -ω^2*sin(u[1])
        end
        # solve problem 
        prob = ODEProblem(nlp!, u0, tspan, ps)
        soln = solve(prob, algo , saveat=dt)
        return Array(soln), soln.t
    end
end

function  GenData(sys::Int, tspan, dt, u0::Vector; algo=Tsit5())
    ps = begin
        if sys==1
            [10., 28., 8/3]
        elseif sys==2
            [0.2, 0.5, 5.7]
        elseif sys==3
            [0.05]
        elseif sys==4 
            [-4., 1]
        elseif sys==5
            [0.8]
        elseif sys==6
            [2.]
        end
    end

    return GenData(sys, tspan, dt, u0, ps; algo=algo)
end