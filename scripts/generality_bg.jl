using DrWatson
@quickactivate "SLIC"
using JLD , Plots, FFTW, LinearAlgebra, StatsBase

# set plot defaults
default(dpi=300, grid=false, fontfamily="computer modern") 

# functions to solve burger's + basic plotting functionality 
include(srcdir("sim_burger.jl"))

# load data 
files = readdir(datadir("sims"))
bgdata = load(datadir("sims", files[5], "burger_results_main.jld"))

# get burger info 
ts = bgdata["ts"][:]
xs = bgdata["xs"][:]
usim = bgdata["Xtrue"]'

# get models 
ind = 4
Ξtrue = bgdata["Ξtrue"][:]
Ξes_slic = mean(bgdata["Ξes_slic"][:,:,:,ind,1], dims=3)[:]
Ξes_aic = mean(bgdata["Ξes_aic"][:,:,:,ind,1], dims=3)[:]

# define params for burgers eqn simulation
c = 1. # 'speed'
ν = 0.1 # diffusion
L = 2*abs(xs[1])
dx = abs(xs[2]- xs[1])
N = length(xs)
dt = ts[2] - ts[1] 
tspan = (ts[1], ts[end])

# freqs 
kappa = (2*pi/L)*(-N/2:N/2-1)
kappa = fftshift(kappa) 

# params for sim 
ptrue = vcat(Ξtrue,kappa)
paic = vcat(Ξes_aic,kappa)
pslic = vcat(Ξes_slic,kappa)
ps = [ptrue, paic, pslic]

# ICs 
u00 = usim[:,1] # from training
W = 70 
u01 = -1*vcat(zeros(Int((N-W)/2)), fill(1,W), zeros(Int((N-W)/2))) # inverted rectangular wave
u02 = @. -2.5*exp(-(xs - xs[170])^2) + 2*exp(-(xs + xs[170])^2) # pair of Gaussians

usol_true0, usol_aic0, usol_slic0 = SimBurg(u00, tspan, dt, ps)
usol_true1, usol_aic1, usol_slic1 = SimBurg(u01, tspan, dt, ps)
usol_true2, usol_aic2, usol_slic2 = SimBurg(u02, tspan, dt, ps)

# plots
p0 = PlotBurg(ts, xs, usol_true0, usol_aic0, usol_slic0)
p1 = PlotBurg(ts, xs, usol_true1, usol_aic1, usol_slic1)
p2 = PlotBurg(ts, xs, usol_true2, usol_aic2, usol_slic2)

display(p0)
display(p1)
display(p2)
