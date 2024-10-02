using DrWatson
@quickactivate "SLIC"
using JLD , Plots

# set plot defaults
default(dpi=300, grid=false, fontfamily="computer modern", framestyle=:box)

# include src dirs 
#include(srcdir("sparse_regress.jl"))
include(srcdir("vis_results.jl"))
include(srcdir("kernel_discovery.jl"))

files = readdir(datadir("exp_raw"))
kmer_data = load(datadir("exp_raw", files[3])) # single cluster data 
kest_data = load(datadir("sims", "KernData.jld")) # kernel es

# load data and scale by 1st data point 
X = kmer_data["data"]
Xn = X/X[1,1] 

# load the data from the sims
inds = kest_data["inds"]
kest = kest_data["kest"]
simdata = kest_data["simdata"]

# plot the fit (optional)
pfit = scatter(0:27, Xn'[:,1:3], label=["Monomer" "Dimer" "Trimer"], ms= 6, legendfontsize=10, 
framestyle=:box, color=[1 2 3])
plot!(0:27, simdata'[:,1:3], color=:red, lw=2.5, label=["Est. kernel" nothing nothing])
ylims!(-0.01, 1.1)
xlims!(-1., 28)
xlabel!("Time (min)")
ylabel!("Normalized k-mers")
display(pfit)

# Now create a concatenated version of the input data. This helps enforce symmetry condition K_ij = K_ji
L = size(inds,2)
gcols = [i for i=1:L if inds[1,i] != inds[2,i]] # get non-repeated indices
inds_cc = hcat(inds, vcat(inds[2,gcols]', inds[1, gcols]')) # flip (i <-> j) and stack 
kest_cc = vcat(kest, kest[gcols]) # concat the kernel

# Define the library 
function RNANPKernLib(inds; df=1.5)
    # get the cluster indices i.e. how many particles are there per kernel 
    i = inds[1,:]
    j = inds[2,:]
        
    # parts of Brownian Kernel 
    kc = ones(length(i))
    kqi = @. (i/j)^(1/df) 
    kqj = @. (j/i)^(1/df) 

    # part of shear & sedimentation
    kshr = @. (i^(1/df) + j^(1/df))^3
    kd = @. abs(i^(1/df)-j^(1/df))

    # other potential linear kernel terms
    kpi = @. i^(1/df) 
    kpj = @. j^(1/df) 

    # Library of terms 
    K = vcat(kc', kd', kpi', kpj', kqi', kqj', kpi'.^3, kpi'.^2 .* kpj', kpj'.^2 .* kpi',  kpj'.^3) 

    # Add them to our model library 
    θ = K[1,:]
    n = size(K,1)

    for i=2:n 
        θ = hcat(θ, K[i,:])
    end
    
    # add |i^1/df - j^1/df|*(all prev kernel components) 
    for i=7:n 
        θ = hcat(θ, kd.*K[i,:])
    end
    
    return θ
end

# Define params for model discovery   
tol = 0.5 # prob thresh 
num_batches1 = 10 # number of ensembles for df optimization loop
num_batches2 = 1000 # number of ensembles for final model discovery
maxiters = 1000 # max iterations for optimization of df 

df_slic , Ξslic , ips_slic = KernDDMD(RNANPKernLib, inds_cc, kest_cc, "slic";
    tol = tol,
    num_batches1 = num_batches1 ,
    num_batches2 = num_batches2 ,
    maxiters = maxiters
)

df_aic , Ξaicc , ips_aic = KernDDMD(RNANPKernLib, inds_cc, kest_cc, "aicc";
    tol = tol,
    num_batches1 = num_batches1 ,
    num_batches2 = num_batches2 ,
    maxiters = maxiters
)

display(VisResults(Ξaicc, "AICc"))
display(VisResults(Ξslic, "SLIC"))

# How well does SLIC predict other experiments?  
# Under constant kernel average cluster size should obey: <N> = 1 + t/τ, t=time, τ=growth rate
# Here we assume 1/τ ∝ Ξslic[1]
# load data: 
rgdata = load(datadir("exp_raw", files[2])) 
orig_rg_avg = rgdata["orig_rg_avg"] # avg cluster vs time data for unmodified RNA-NP formulation
rev_rg_avg = rgdata["rev_rg_avg"] # avg cluster vs time data for modified RNA-NP formulation (2x dilution)

# fits 
tp = 0:30
b1, a1 =  [ones(31) tp*Ξslic[1]]\orig_rg_avg 
b2 = ones(31)\(rev_rg_avg - tp*Ξslic[1]*a1/2) # just fit the intercept and use 1/2 of the slope of original for concentration change

# plot 
pnp = scatter(tp, orig_rg_avg, color=1, label="Exp: Original", ms=6)
plot!(tp, b1 .+ tp*Ξslic[1]*a1, label="Pred: Original", color=:blue, lw=3)
scatter!(tp, rev_rg_avg, label="Exp: 2x dilution", color=2, ms=6)
plot!(tp, b2 .+ tp*Ξslic[1]*a1/2, label="Pred: 2x dilution", color=:orange, lw=3)
plot!(legend=:topleft)
xlabel!("Time (min)")
xlims!(-1,31)
ylims!(0.8,6)
ylabel!("Avg Particles/cluster")
display(pnp)