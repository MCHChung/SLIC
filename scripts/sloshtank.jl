using DrWatson
@quickactivate "SLIC"
using MAT , Plots

# NOTE: if you want to display a plot p, just use display(p). Lines that save plots are commented out here. 
# NOTE: the first time running this script, there may be some odd precompilation messages due to having to use older packages to avoid conflicts

# set plot defaults
default(dpi=300, grid=false, fontfamily="computer modern")

# include src dirs 
include(srcdir("derivative.jl"))
include(srcdir("galerkin_proj.jl"))  
include(srcdir("sparse_regress.jl")) # performs model discovery
include(srcdir("sim_ode.jl")) # simulates ode given model 
include(srcdir("vis_results.jl")) # visualizes output model coefficient matrix
include(srcdir("freq_resp_curves.jl")) # plots frequency response curves

# get data, originally from ==> https://www.nature.com/articles/s41467-022-28518-y ,  https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/phase-lag-predicts-nonlinear-response-maxima-in-liquidsloshing-experiments/2CD68A3A9A3AFC648B311D6C01835988 
data = matread(datadir("exp_raw", "SloshingData.mat"))

# load unforced training data
ts = data["xData"][1,1]
dt = ts[2] - ts[1]
u = data["xData"][1,2]
tsteps, u, du = DataWithFirstDeriv(u[:]', ts[:], dt) 
X = vcat(u, du)

# define library using galerkin projection
function TankLib(X, ts, wind; p=4, Δ=1)
    θ = wInt_sc(X[1,:], ts, wind, p=p)
    for i=2:size(X,1)
        θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p))
    end

    for i=1:size(X,1)
        for j=i:size(X,1)
            θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p))
        end
    end
    
    for i=1:size(X,1)
        for j=i:size(X,1)
            for k=j:size(X,1)
                θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p))
            end
        end
    end
    
    return θ 
end
# this is the same version of the above without galerkin proj, for forecasting
function TankLib(X)
    θ = X[1,:]
    for i=2:size(X,1)
        θ  = hcat(θ, X[i,:])
    end

    for i=1:size(X,1)
        for j=i:size(X,1)
            θ = hcat(θ, X[i,:].*X[j,:])
        end
    end

    for i=1:size(X,1)
        for j=i:size(X,1)
            for k=j:size(X,1)
                θ = hcat(θ, X[i,:].*X[j,:].*X[k,:])
            end
        end
    end
    
    return θ
end

p=10
_,_,wind = FindW(X, tsteps, ws=21:2:55, p=p)

train_ind = 300 # final time point for training data 
θtank = TankLib(X[:, 1:train_ind], tsteps[1:train_ind], wind, p=p)
qt = dwInt_sc(X[:, 1:train_ind], tsteps[1:train_ind], wind, p=p)

# model discovery
tol = 0.7
num_batches = 250
trainpct = 60 
Ξslic, _ = EnAdSR(θtank, qt', "slic", tol=tol, num_batches=num_batches, trainpct=trainpct)
Ξaic, _ = EnAdSR(θtank, qt', "aic", tol=tol, num_batches=num_batches, trainpct=trainpct)
Ξaicc, _ = EnAdSR(θtank, qt', "aicc", tol=tol, num_batches=num_batches, trainpct=trainpct)
Ξhqic, _ = EnAdSR(θtank, qt', "hqic", tol=tol, num_batches=num_batches, trainpct=trainpct)
Ξbic, _ = EnAdSR(θtank, qt', "bic", tol=tol, num_batches=num_batches, trainpct=trainpct)
Ξkic, _ = EnAdSR(θtank, qt', "kic", tol=tol, num_batches=num_batches, trainpct=trainpct)
Ξbc, _ = EnAdSR(θtank, qt', "bc", tol=tol, num_batches=num_batches, trainpct=trainpct)
Ξexp = [0. 1. 0. 0. 0. 0. 0. 0. 0. ; -7.8^2 -2*0.065 0. 0. 0. 0.36 0. 0. 0.]' # experimental model from 2nd hyperlink above

# Results
# make heatmap of log of the magnitudes of models
lΞexp = log10.(abs.(Ξexp[:,2:2])) 
clims = (-2, 2)
pmodel_aic = VisResults(lΞexp, log10.(abs.(Ξaic[:,2:2])) , "AIC", clims=clims)
pmodel_aicc = VisResults(lΞexp, log10.(abs.(Ξaicc[:,2:2])), "AICc", clims=clims)
pmodel_hqic = VisResults(lΞexp, log10.(abs.(Ξhqic[:,2:2])), "HQIC", clims=clims)
pmodel_bic = VisResults(lΞexp, log10.(abs.(Ξbic[:,2:2])), "BIC", clims=clims)
pmodel_kic = VisResults(lΞexp, log10.(abs.(Ξkic[:,2:2])), "KIC", clims=clims)
pmodel_bc = VisResults(lΞexp, log10.(abs.(Ξbc[:,2:2])), "BC", clims=clims)
pmodel_slic = VisResults(lΞexp, log10.(abs.(Ξslic[:,2:2])), "SLIC", clims=clims)

display(pmodel_aic)
display(pmodel_aicc)
display(pmodel_hqic)
display(pmodel_bic)
display(pmodel_kic)
display(pmodel_bc)
display(pmodel_slic)

#=
# save plots 
wsave(plotsdir("main", "unforced_AIC_model.png"), pmodel_aic)
wsave(plotsdir("main","unforced_AICc_model.png"), pmodel_aicc)
wsave(plotsdir("main","unforced_HQIC_model.png"), pmodel_hqic)
wsave(plotsdir("main","unforced_BIC_model.png"), pmodel_bic)
wsave(plotsdir("main","unforced_KIC_model.png"), pmodel_kic)
wsave(plotsdir("main","unforced_BC_model.png"), pmodel_bc)
wsave(plotsdir("main","unforced_SLIC_model.png"), pmodel_slic)
=#  

# forecast and visualize
Xpred = SimODE(TankLib, Ξslic, X[:,1], tsteps)
# entire plot 
p1 = plot(tsteps, X[1,:], color=:blue, label="Exp.")
plot!(tsteps, Xpred[1,:], color=:red, ls=:dash, label="Pred.")
vline!([tsteps[train_ind]], color=:black, label="Final train pt.")
xlabel!("Time (s)")
ylabel!("Center of Mass/Tank Width")

# zoomed in view
p2 = plot(tsteps[200:700], X[1,200:700], color=:blue, label="Exp.")
plot!(tsteps[200:700], Xpred[1,200:700], color=:red, ls=:dash, label="Pred.")
title!("Zoomed-in version of above:")
xlabel!("Time (s)")
ylabel!("Center of Mass/Tank Width")

p = plot(p1,p2, layout=(2,1), size=(800,800))
display(p)
#wsave(plotsdir("unforced_sloshtank_slic_forecast.png"), p)

# Now, using SLIC model, plot the freq-resp curves 
# If there are weird precompilation messages the first time running this, apologies.
# Ω = driving freq 
# ω1 = fundamental frequency 
p_amp, p_phase = PlotFreqRespCurves(Ξslic)
display(p_amp)
display(p_phase)

#wsave(plotsdir("forced_sloshtank_slic_amp_pred.png"), p_amp)
#wsave(plotsdir("forced_sloshtank_slic_phase_pred.png"), p_phase)