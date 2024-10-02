using DrWatson
@quickactivate "SLIC"

# This file reproduces the plots from Fig 2a,b of the main text 

# get performance metrics 
include(srcdir("performance_metrics.jl"))

# plot settings
default(dpi=300, fontfamily="computer modern", grid=false)

# load ODE results
files = readdir(datadir("sims", "ode_results_main")) # all simulated ODE data 
lordata = load(datadir("sims", "ode_results_main", files[2]))
rossdata = load(datadir("sims", "ode_results_main", files[5]))
lvdata = load(datadir("sims", "ode_results_main", files[3]))
brusdata = load(datadir("sims", "ode_results_main", files[1]))
vdpdata = load(datadir("sims", "ode_results_main", files[6]))
nlpdata = load(datadir("sims", "ode_results_main", files[4]))


# to get accurate metrics for VdP, we duplicate the 2nd column. 1st column (dx/dt = v) is a trivial prediction
Ξtrue_vdp_dp = similar(vdpdata["Ξtrue"]) 
Ξtrue_vdp_dp[:,1] = Ξtrue_vdp_dp[:,2] = vdpdata["Ξtrue"][:,2]

Ξes_slic_vdp_dp = similar(vdpdata["Ξes_slic"][:,:,:,:,1])
Ξes_aic_vdp_dp = similar(vdpdata["Ξes_aic"][:,:,:,:,1])

Ξes_slic_vdp_dp[:, 1, :, :] = Ξes_slic_vdp_dp[:, 2, :, :] = vdpdata["Ξes_slic"][:,2,:,:,1]
Ξes_aic_vdp_dp[:, 1, :, :] = Ξes_aic_vdp_dp[:, 2, :, :] = vdpdata["Ξes_aic"][:,2,:,:,1]

# get model outputs
Ξtrues = [lordata["Ξtrue"], rossdata["Ξtrue"],lvdata["Ξtrue"],brusdata["Ξtrue"], Ξtrue_vdp_dp, nlpdata["Ξtrue"]] ;
Ξes_slic = [lordata["Ξes_slic"][:,:,:,:,1], rossdata["Ξes_slic"][:,:,:,:,1],lvdata["Ξes_slic"][:,:,:,:,1],brusdata["Ξes_slic"][:,:,:,:,1],Ξes_slic_vdp_dp, nlpdata["Ξes_slic"][:,:,:,:,1]] ; 
Ξes_aicc = [lordata["Ξes_aic"][:,:,:,:,1], rossdata["Ξes_aic"][:,:,:,:,1],lvdata["Ξes_aic"][:,:,:,:,1],brusdata["Ξes_aic"][:,:,:,:,1],Ξes_aic_vdp_dp, nlpdata["Ξes_aic"][:,:,:,:,1]] ; 

# Note: the final index = 1 is the 'downsampling' index, here indicating we are not downsampling 

# plots 
fpr_ylims = 1.5*ones(6)
fpr_ylims[4] = 1.75
p_err, p_acc, p_fpr = PerformancePlots(Ξtrues , Ξes_slic, Ξes_aicc, lw=4, ms=6, fpr_ylims =fpr_ylims)
display(p_acc)
display(p_fpr)