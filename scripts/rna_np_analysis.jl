using DrWatson
@quickactivate "SLIC"
using JLD , Plots

# NOTE: if you want to display a plot p, just use display(p). Lines that save plots are commented out here. 

# set plot defaults
default(dpi=300, grid=false, fontfamily="computer modern", framestyle=:box)
#scalefontsizes(1.2^2)

# include src dirs 
include(srcdir("vis_results.jl")) # visualization
include(srcdir("sim_np_agg.jl")) # simulate aggregation
include(srcdir("sparse_regress.jl")) # for scoring function

# get data 
kmer_data = load(datadir("exp_raw", "MaxEntNPData.jld")) # single cluster data 
kest_data = load(datadir("sims", "KernData.jld")) # kernel es
np_models = load(datadir("rna_np_models.jld")) # models extracted by SLIC from "rna_np_models.jl"

# library
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

# get best model from random initializations for each IC
ic_labels = ["aic", "aicc", "hqic", "bic", "kic", "bc", "slic"]
Ξbest = [] # holds model with best score 
ibest = [] # holds index for best score
y = kest_data["kest"][:,1:1]
inds = kest_data["inds"]
for ic in ic_labels
    score_ic_best = Inf
    ind_best = 0
    for i=1:3
        θ = RNANPKernLib(inds, df=np_models[ic][i][1])
        score_ic = score(y, θ, np_models[ic][i][2], ic, 0)
        if score_ic < score_ic_best
            score_ic_best = score_ic
            ind_best = i
        end
    end
    push!(Ξbest, np_models[ic][ind_best][2])
    push!(ibest, ind_best)
end


# visualize results and plot
clims = (-5e-3, 5e-2)
pmodel_aic = VisResults(Ξbest[1], "AIC", clims=clims)
pmodel_aicc = VisResults(Ξbest[2], "AICc", clims=clims)
pmodel_hqic = VisResults(Ξbest[3], "HQIC", clims=clims)
pmodel_bic = VisResults(Ξbest[4], "BIC", clims=clims)
pmodel_kic = VisResults(Ξbest[5], "KIC", clims=clims)
pmodel_bc = VisResults(Ξbest[6], "BC", clims=clims)
pmodel_slic = VisResults(Ξbest[7], "SLIC", clims=clims)
# experimental model from https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202414305 
Ξexp = vcat(0.032, zeros(length(Ξbest[7])-1))[:, 1:1] 
pmodel_exp = VisResults(Ξexp, "Exp.", clims=clims)

# visualization
display(pmodel_aic)
display(pmodel_aicc)
display(pmodel_bc)
display(pmodel_hqic)
display(pmodel_bic)
display(pmodel_kic)
display(pmodel_slic)
display(pmodel_exp)

#=
# save visualizations
wsave(plotsdir("main", "rna-np_model_aic.png"), pmodel_aic)
wsave(plotsdir("main","rna-np_model_aicc.png"), pmodel_aicc)
wsave(plotsdir("main","rna-np_model_hqic.png"), pmodel_hqic)
wsave(plotsdir("main","rna-np_model_bic.png"), pmodel_bic)
wsave(plotsdir("main","rna-np_model_kic.png"), pmodel_kic)
wsave(plotsdir("main","rna-np_model_bc.png"), pmodel_bc)
wsave(plotsdir("main","rna-np_model_slic.png"), pmodel_slic)
wsave(plotsdir("main","rna-np_model_exp.png"), pmodel_exp)
=# 

# simulate aggregation 
ind = 7 # slic
θ = RNANPKernLib(inds, df=np_models[ic_labels[ind]][ibest[ind]][1])
N = 50
kest = θ * Ξbest[ind]
Xn = kmer_data["Ns"][1:N ,:]
Xn = Xn/Xn[1,1]
tsteps, Xest = SimAgg(N, Xn[:, 1], vec(kest))

# plot prediction
labels = ["Monomer", "Dimer", "Trimer"]
exp_colors = [1,2,3]
pred_colors = [:blue, :orange, :green]
pfit = plot()
for i=1:length(labels)
    scatter!(0:27, Xn'[:,i], label="Exp: "*labels[i], ms= 7, legendfontsize=10, 
    framestyle=:box, color=exp_colors[i])
    plot!(0:27, Xest'[:,i], color=pred_colors[i], label="Pred: "*labels[i], lw = 5)
    ylims!(-0.01, 1.1)
    xlims!(-1., 28)
end
xlabel!("Time (min)")
ylabel!("Normalized k-mers")
display(pfit)
#wsave(plotsdir("main","rna-np_slic_kmer_exp_pred.png"), pfit)

# How well does SLIC predict other experiments?  
# Under constant kernel average cluster size should obey: <N> = 1 + t/τ, t=time, τ=growth rate
# Here we assume 1/τ ∝ Ξslic[1]
# load data: 
rgdata = load(datadir("exp_raw", "RNA-NP_AvgClusterData.jld")) 
orig_rg_avg = rgdata["orig_rg_avg"] # avg cluster vs time data for unmodified RNA-NP formulation
rev_rg_avg = rgdata["rev_rg_avg"] # avg cluster vs time data for modified RNA-NP formulation (2x dilution)

# fits 
tp = 0:30
b1, a1 =  [ones(31) tp*Ξbest[7][1]]\orig_rg_avg 
b2 = ones(31)\(rev_rg_avg - tp*Ξbest[7][1]*a1/2) # just fit the intercept and use 1/2 of the slope of original for concentration change

# plot 
pnp = scatter(tp, orig_rg_avg, color=1, label="Exp: Original", ms=7)
plot!(tp, b1 .+ tp*Ξbest[7][1]*a1, label="Pred: Original", color=:blue, lw=5)
scatter!(tp, rev_rg_avg, label="Exp: 2x dilution", color=2, ms=7)
plot!(tp, b2 .+ tp*Ξbest[7][1]*a1/2, label="Pred: 2x dilution", color=:orange, lw=5)
plot!(legend=:topleft)
xlabel!("Time (min)")
xlims!(-1,31)
ylims!(0.8,6)
ylabel!("Avg Particles/cluster")
#wsave(plotsdir("main","rna-np_slic_new_exp_pred.png"), pnp)
