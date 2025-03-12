using DrWatson
@quickactivate "SLIC"
using JLD , Plots

# The purpose of this script to generate estimated kernel data that we will use for model discovery 
# The results can vary based on initial estimate, so we save a good result fron this script   

# set plot defaults
default(dpi=300, grid=false, fontfamily="computer modern")

# include src dirs 
include(srcdir("est_kernel.jl"))
include(srcdir("smooth.jl"))

kmer_data = load(datadir("exp_raw", "MaxEntNPData.jld"))

# load data and normalize by 1st data point 
ts = kmer_data["ts"]
N = 50
X = kmer_data["Ns"][1:N, :]
Xn = X / X[1,1] 
Xsm = smooth_ode(Xn)

# plot 
p = scatter(ts, Xsm'[:,1:3], label=["Monomer" "Dimer" "Trimer"], ms= 6, legendfontsize=10, 
framestyle=:box, color=[1 2 3])
#ylims!(-0.01, 1.1)
xlims!(-1., 28)
xlabel!("Time (min)")
ylabel!("Normalized k-mers")
display(p)

# estimate kernel 
#N = size(X,1) 
u0vec = Xsm[:,1]
k0 = [] # use random initial vec 
#k0 = 1e-2*ones(N)

ts, inds, k0, kest, simdata = EstimKernel(N, Xsm, u0vec, k0, tspan = (0.0, 27.0), use_rand_k0=true,
    lb_mag=1e-3, ub_mag = 1e-1)
  
# * IMPORTANT! Sometimes the solution is clearly a bad fit. Need to inspect, unfortunately *
pfit = scatter(0:27, Xsm'[:,1:3], label=["Monomer" "Dimer" "Trimer"], ms= 6, legendfontsize=10, 
framestyle=:box, color=[1 2 3])
plot!(0:27, simdata'[:,1:3], color=:red, lw=2.5, legend=false)
#ylims!(-0.01, 1.1)
xlims!(-1., 28)
xlabel!("Time (min)")
ylabel!("Normalized k-mers")
display(pfit)
 


# Here we save a good fit after inspection, currently commented out
KernData = Dict(
    "ts" => ts, 
    "inds" => inds,
    "k0" => k0[:], 
    "kest" => kest[:], 
    "simdata" => simdata
)

#wsave(datadir("sims", "KernData.jld"), KernData)
 