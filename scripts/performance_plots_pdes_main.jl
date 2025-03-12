using DrWatson
@quickactivate "SLIC"

# This file reproduces the plots from Fig 3a,b of the main text 

# get performance metrics 
include(srcdir("performance_metrics.jl"))

# plot settings
default(dpi=300, fontfamily="computer modern", grid=false)

# load PDE results
bgdata = load(datadir("sims", "pde_results_main", "burger_results_main.jld"))
kdvdata = load(datadir("sims", "pde_results_main", "kdv_results_main.jld"))
ksdata = load(datadir("sims", "pde_results_main","ks_results_main.jld"))
nlsdata = load(datadir("sims", "pde_results_main", "nls_results_main.jld"))
sgdata = load(datadir("sims", "pde_results_main", "sg_results_main.jld"))

# get models 
labels = ["Ξslics", "Ξaics", "Ξaiccs", "Ξbics", "Ξhqics", "Ξkics", "Ξbcs"]
Ξtrues = [bgdata["Ξtrue"], kdvdata["Ξtrue"], ksdata["Ξtrue"], nlsdata["Ξtrue"], sgdata["Ξtrue"]] ;
Ξs = []
for ic=1:length(labels)
    Ξ = [bgdata[labels[ic]][:,:,:,:,1], kdvdata[labels[ic]][:,:,:,:,1], ksdata[labels[ic]][:,:,:,:,1], nlsdata[labels[ic]][:,:,:,:,1], sgdata[labels[ic]][:,:,:,:,1]] ;
    push!(Ξs, Ξ)
end

# make plots
fpr_ylims=1.1*ones(5)
err_ylims = [(1e-9,1e0), (1e-7,1e0), (1e-6,1e0), (1e-9,1e0),(1e-5,1e0)]
#p_err, p_acc, p_fpr = PerformancePlots(Ξtrues , Ξes_slic, Ξes_bic, lw=4, ms=6, labels = ["Burger" , "KdV", "K-S" , "NLS", "S-G"], fpr_ylims=fpr_ylims)
p_err, p_acc, p_fpr = PerformancePlots(Ξtrues , Ξs,  lw=5, ms=7, 
labels = ["Burger" , "KdV", "K-S" , "NLS", "S-G"], fpr_ylims=fpr_ylims, err_ylims=err_ylims, alpha=0.7)

#=
wsave(plotsdir("main", "sim_pde_acc.png"), p_acc)
wsave(plotsdir("main", "sim_pde_fpr.png"), p_fpr)
wsave(plotsdir("main", "sim_pde_err.png"), p_err)
=# 

display(p_acc)
display(p_fpr)
display(p_err)