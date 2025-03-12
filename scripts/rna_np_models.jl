using DrWatson
@quickactivate "SLIC"
using JLD , Plots

# set plot defaults
default(dpi=300, grid=false, fontfamily="computer modern", framestyle=:box)

# include src dirs 
#include(srcdir("sparse_regress.jl"))
include(srcdir("vis_results.jl"))
include(srcdir("kernel_discovery.jl"))
include(srcdir("sim_np_agg.jl"))

kmer_data = load(datadir("exp_raw", "MaxEntNPData.jld")) # single cluster data 
kest_data = load(datadir("sims", "KernData.jld")) # kernel es

# load data and scale by 1st data point 
X = kmer_data["Ns"]
N = 50
Xn = X[1:N,:]/X[1,1] 

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
#wsave(plotsdir("kmer_fit.png"), pfit)

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
num_batches1 = 100 # number of ensembles for df optimization loop
num_batches2 = 1000 # number of ensembles for final model discovery
maxiters = 1000 # max iterations for optimization of df 
trainpct = 80

aic_results = []
aicc_results = []
hqic_results = []
bic_results = []
kic_results = []
bc_results = []
slic_results = []

for i=1:3
    df_slic , Ξslic , ips_slic = KernDDMD(RNANPKernLib, inds_cc, kest_cc, "slic";
        tol = tol,
        num_batches1 = num_batches1 ,
        num_batches2 = num_batches2 ,
        maxiters = maxiters,
        trainpct=trainpct
    )
    push!(slic_results, (df_slic , Ξslic , ips_slic))

    df_aic , Ξaic , ips_aic = KernDDMD(RNANPKernLib, inds_cc, kest_cc, "aic";
        tol = tol,
        num_batches1 = num_batches1 ,
        num_batches2 = num_batches2 ,
        maxiters = maxiters,
        trainpct=trainpct
    )
    push!(aic_results, (df_aic , Ξaic , ips_aic))

    df_aicc , Ξaicc , ips_aicc = KernDDMD(RNANPKernLib, inds_cc, kest_cc, "aicc";
        tol = tol,
        num_batches1 = num_batches1 ,
        num_batches2 = num_batches2 ,
        maxiters = maxiters,
        trainpct=trainpct
    )
    push!(aicc_results, (df_aicc , Ξaicc , ips_aicc))

    df_hqic , Ξhqic , ips_hqic = KernDDMD(RNANPKernLib, inds_cc, kest_cc, "hqic";
        tol = tol,
        num_batches1 = num_batches1 ,
        num_batches2 = num_batches2 ,
        maxiters = maxiters,
        trainpct=trainpct
    )
    push!(hqic_results, (df_hqic , Ξhqic , ips_hqic))

    df_bic , Ξbic , ips_bic = KernDDMD(RNANPKernLib, inds_cc, kest_cc, "bic";
        tol = tol,
        num_batches1 = num_batches1 ,
        num_batches2 = num_batches2 ,
        maxiters = maxiters,
        trainpct=trainpct
    )
    push!(bic_results, (df_bic , Ξbic , ips_bic))

    df_kic , Ξkic , ips_kic = KernDDMD(RNANPKernLib, inds_cc, kest_cc, "kic";
        tol = tol,
        num_batches1 = num_batches1 ,
        num_batches2 = num_batches2 ,
        maxiters = maxiters,
        trainpct=trainpct
    )
    push!(kic_results, (df_kic , Ξkic , ips_kic))

    df_bc , Ξbc , ips_bc = KernDDMD(RNANPKernLib, inds_cc, kest_cc, "bc";
        tol = tol,
        num_batches1 = num_batches1 ,
        num_batches2 = num_batches2 ,
        maxiters = maxiters,
        trainpct=trainpct
    )
    push!(bc_results, (df_bc , Ξbc , ips_bc))

end

rna_np_models = Dict(
    "aic" => aic_results,
    "aicc" => aicc_results,
    "hqic" => hqic_results,
    "bic" => bic_results,
    "kic" => kic_results,
    "bc" => bc_results,
    "slic" => slic_results,
)

#wsave(datadir("rna_np_models.jld"), rna_np_models)