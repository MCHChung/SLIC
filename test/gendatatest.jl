using DrWatson, Test
@quickactivate "SLIC"

include(srcdir("generate_ode_data.jl"))
sys = 1
tspan = (0., 10.)
dt = 1e-3
u0 = [-8., -7., 27]
ps = [10.,28.,8/3]
GenData(sys, tspan, dt, u0, ps)