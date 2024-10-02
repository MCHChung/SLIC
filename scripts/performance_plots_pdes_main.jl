using DrWatson
@quickactivate "SLIC"

# This file reproduces the plots from Fig 3a,b of the main text 

# get performance metrics 
include(srcdir("performance_metrics.jl"))

# plot settings
default(dpi=300, fontfamily="computer modern", grid=false)

# load PDE results
files = readdir(datadir("sims", "pde_results_main"))
bgdata = load(datadir("sims", "pde_results_main", files[1]))
kdvdata = load(datadir("sims", "pde_results_main", files[2]))
ksdata = load(datadir("sims", "pde_results_main", files[3]))
nlsdata = load(datadir("sims", "pde_results_main", files[4]))
sgdata = load(datadir("sims", "pde_results_main", files[5]))

# get models 
Ξtrues = [bgdata["Ξtrue"], kdvdata["Ξtrue"], ksdata["Ξtrue"], nlsdata["Ξtrue"], sgdata["Ξtrue"]] ;
Ξes_slic = [bgdata["Ξes_slic"][:,:,:,:,1], kdvdata["Ξes_slic"][:,:,:,:,1], ksdata["Ξes_slic"][:,:,:,:,1], nlsdata["Ξes_slic"][:,:,:,:,1], sgdata["Ξes_slic"][:,:,:,:,1]] ;
Ξes_aicc = [bgdata["Ξes_aic"][:,:,:,:,1], kdvdata["Ξes_aic"][:,:,:,:,1], ksdata["Ξes_aic"][:,:,:,:,1] , nlsdata["Ξes_aic"][:,:,:,:,1], sgdata["Ξes_aic"][:,:,:,:,1]]; 

# make plots
fpr_ylims=1.5*ones(5)
fpr_ylims[1] = 1.75
p_err, p_acc, p_fpr = PerformancePlots(Ξtrues , Ξes_slic, Ξes_aicc, lw=4, ms=6, labels = ["Burger" , "KdV", "K-S" , "NLS", "S-G"], fpr_ylims=fpr_ylims)
display(p_acc)
display(p_fpr)