using DrWatson
@quickactivate "SLIC"
using JLD, Random

# This script generates the data used for model discovery in the main text

# functions to extract models and visualize results
include(srcdir("generate_ode_data.jl"))

# set seed 
rng = Random.default_rng()
Random.seed!(rng, 0)

# =================== Lorenz ===================

# timestep details 
dt = 1e-3 
ts = 0.:dt:10.
tspan = (ts[1] , ts[end])
# generate data 
Xtrues = []
for i=1:3
    u0 = rand(rng, -50:50 , 3)
    Xtrue , _ = GenData(1, tspan, dt , u0)
    push!(Xtrues, Xtrue)
end

# define true model and save 
Ξtrue =  [-10. 10. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.; 28. -1. 0. 0. 0. -1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.; 0. 0. -8/3 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]'
lordata = Dict(
    "Ξtrue" => Ξtrue, 
    "Xtrues" => Xtrues, 
    "ts" => ts
)
#wsave(datadir("sims", "ode_data", "lordata.jld"), lordata)


# =================== Rossler =================== 

# timestep details 
dt = 1e-3
ts = 0.:dt:10.
tspan = (ts[1] , ts[end])
# generate data 
Xtrues = []
for i=1:3
    u0 = rand(rng, 0:5 , 3)
    Xtrue , _ = GenData(2, tspan, dt , u0)
    push!(Xtrues, Xtrue)
end

# define model and save 
Ξtrue = [0.0 0.0 0.5; 0.0 1.0 0.0; -1.0 0.2 0.0; -1.0 0.0 -5.7; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
rossdata = Dict(
    "Ξtrue" => Ξtrue, 
    "Xtrues" => Xtrues, 
    "ts" => ts
)
#wsave(datadir("sims", "ode_data", "rossdata.jld"), rossdata)

# =================== L-V ===================

# timestep details 
dt = 1e-2
ts = 0.:dt:20.
tspan = (ts[1] , ts[end])
# generate data 
Xtrues = []
for i=1:3
    u0 = rand(rng, 1:50 , 2)
    Xtrue , _ = GenData(3, tspan, dt , u0)
    push!(Xtrues, Xtrue)
end

Ξtrue = [1.0 0.0; 0.0 -1.0; 0.0 0.0; -0.05 0.05; 0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0]
lvdata = Dict(
    "Ξtrue" => Ξtrue, 
    "Xtrues" => Xtrues, 
    "ts" => ts
)
#wsave(datadir("sims", "ode_data", "lvdata.jld"), lvdata)

# ================= Brusselator =================== 

# timestep details 
dt = 1e-3
ts = 0.:dt:10.
tspan = (ts[1] , ts[end])
# generate data 
Xtrues = []
for i=1:3
    u0 = rand(rng, 0.:10.)
    u0 = [0., u0]
    Xtrue , _ = GenData(4, tspan, dt , u0)
    push!(Xtrues, Xtrue)
end

Ξtrue = [1.0 0.0; -4.0 3.0; 0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0; 1.0 -1.0; 0.0 0.0; 0.0 0.0]
brusdata = Dict(
    "Ξtrue" => Ξtrue, 
    "Xtrues" => Xtrues, 
    "ts" => ts
)
#wsave(datadir("sims", "ode_data", "brusdata.jld"), brusdata)

# =================== VdP ===================

# timestep details 
dt = 1e-2
ts = 0.:dt:30.
tspan = (ts[1] , ts[end])

# generate data 
Xtrues = []
for i=1:3
    u0 = rand(rng, 0:0.5:3 , 2)
    Xtrue , _ = GenData(5, tspan, dt , u0)
    push!(Xtrues, Xtrue)
end

Ξtrue = [0.0 -1.0; 1.0 0.8; 0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 -0.8; 0.0 0.0; 0.0 0.0]
vdpdata = Dict(
    "Ξtrue" => Ξtrue, 
    "Xtrues" => Xtrues, 
    "ts" => ts
)
#wsave(datadir("sims", "ode_data", "vdpdata.jld"), vdpdata)

# =================== NLP ===================

# timestep details 
dt = 1e-2
ts = 0.:dt:20.
tspan = (ts[1] , ts[end])

# generate data 
Xtrues = []
for i=1:3
    u0 = rand(rng, π/2:0.1:3π/4)
    u0 = [u0, 0.]
    Xtrue , _ = GenData(6, tspan, dt , u0)
    push!(Xtrues, Xtrue)
end

Ξtrue = [0.0; -4.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0;;]
nlpdata = Dict(
    "Ξtrue" => Ξtrue, 
    "Xtrues" => Xtrues, 
    "ts" => ts
)
#wsave(datadir("sims", "ode_data", "nlpdata.jld"), nlpdata)