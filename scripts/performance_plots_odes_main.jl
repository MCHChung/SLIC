using DrWatson
@quickactivate "SLIC"

# This file reproduces the plots from Fig 2 of the main text 

# get performance metrics 
include(srcdir("performance_metrics.jl"))

# plot settings
default(dpi=300, fontfamily="computer modern", grid=false)

# load ODE results
lordata = load(datadir("sims", "ode_results_main", "lor_results_main.jld"))
rossdata = load(datadir("sims", "ode_results_main", "ross_results_main.jld"))
lvdata = load(datadir("sims", "ode_results_main", "lv_results_main.jld"))
brusdata = load(datadir("sims", "ode_results_main", "brus_results_main.jld"))
vdpdata = load(datadir("sims", "ode_results_main", "vdp_results_main.jld"))
nlpdata = load(datadir("sims", "ode_results_main", "nlp_results_main.jld"))

# get models 
labels = ["Ξslics", "Ξaics", "Ξaiccs", "Ξbics", "Ξhqics", "Ξkics", "Ξbcs"]
Ξtrue_vdp_dp = similar(vdpdata["Ξtrue"]) 
# to get accurate metrics for VdP, we duplicate the 2nd column. 1st column (dx/dt = v) is a trivial prediction
Ξtrue_vdp_dp[:,1] = Ξtrue_vdp_dp[:,2] = vdpdata["Ξtrue"][:,2]
Ξtrues = [lordata["Ξtrue"], rossdata["Ξtrue"], lvdata["Ξtrue"], brusdata["Ξtrue"], Ξtrue_vdp_dp, nlpdata["Ξtrue"]] ;
Ξs = []
for ic=1:length(labels)
    # to get accurate metrics for VdP, we duplicate the 2nd column. 1st column (dx/dt = v) is a trivial prediction
    Ξs_vdp_dp = similar(vdpdata[labels[ic]][:,:,:,:,1])
    Ξs_vdp_dp[:, 1, :, :] = Ξs_vdp_dp[:, 2, :, :] = vdpdata[labels[ic]][:,2,:,:,1]

    Ξ = [lordata[labels[ic]][:,:,:,:,1], rossdata[labels[ic]][:,:,:,:,1], lvdata[labels[ic]][:,:,:,:,1], 
    brusdata[labels[ic]][:,:,:,:,1], Ξs_vdp_dp, nlpdata[labels[ic]][:,:,:,:,1]] ;
    push!(Ξs, Ξ)
end

# scalefontsizes(1.2^2)
# make plots 
fpr_ylims = 1.1*ones(6)
err_ylims = [(1e-4,1e0), (1e-3,1e0), (5e-5,1e0), (1e-4,1e1),(0.9e-4,1e0), (1e-4,1e2)]
#p_err, p_acc, p_fpr = PerformancePlots(Ξtrues , Ξes_slic, Ξes_aicc, lw=4, ms=6, fpr_ylims =fpr_ylims)
p_err, p_acc, p_fpr, p_leg = PerformancePlots(Ξtrues , Ξs,  lw=5,
 ms=7, fpr_ylims =fpr_ylims, err_ylims=err_ylims, alpha=0.7, legend_on=false)

#=
wsave(plotsdir("main", "sim_ode_acc.png"), p_acc)
wsave(plotsdir("main","sim_ode_fpr.png"), p_fpr)
wsave(plotsdir("main","sim_ode_err.png"), p_err)
wsave(plotsdir("plot_legend.png"), p_leg) # just makes plot w/legend 
=# 

display(p_acc)
display(p_fpr)
display(p_err)
display(p_leg)
