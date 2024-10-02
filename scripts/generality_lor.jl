using DrWatson
@quickactivate "SLIC"
using JLD , Plots, StatsBase, LinearAlgebra, Measures

# set plot defaults
default(dpi=300, grid=false, fontfamily="computer modern", framestyle=:default) 
#scalefontsizes(1.2^4)

# simulates odes given a model (Ξ)
include(srcdir("sim_ode.jl"))

# load data 
files = readdir(datadir("sims"))
lvdata = load(datadir("sims", files[2], "lv_results_main.jld"))

# define LV library 
function LVLib(X)
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

# get true Lotka-Volterra data
ts = lvdata["ts"]
Xlor = lvdata["Xtrue"]
u0 = Xlor[:,1] 
# get avergae L-V models from AICc and SLIC, respectively 
ind = 4
Ξes_aic_avg = mean(lvdata["Ξes_aic"][:,:,:,ind,1], dims=3)[:,:]
Ξes_slic_avg = mean(lvdata["Ξes_slic"][:,:,:,ind,1], dims=3)[:,:]
# get predictions using models from training data 
tspan = (0.,50.)
dt = 0.01
Xtrue = SimODE(LVLib, lvdata["Ξtrue"][:,:], u0, tspan, dt)
Xpred_aic = SimODE(LVLib, Ξes_aic_avg, u0, tspan, dt)
Xpred_slic = SimODE(LVLib, Ξes_slic_avg, u0, tspan, dt)


# Now make predictions from new initial conditions
# simulates new ics for ODE
# Lib: library used to compute deriv: dudt = Lib(u)*Ξ 
# Ξ: model 
# u0s: initial cond s
# tspan: range of time for sim
# dt: time step
function SimNewICs(Lib::Function, Ξs, u0s::AbstractMatrix, tspan, dt)
    Ξtrue = Ξs[1]
    Ξaic = Ξs[2]
    Ξslic = Ξs[3]

    Xtrues = []
    Xaics = []
    Xslics = []
    for i=1:size(u0s,2)
        u0 = u0s[:,i]
        Xtrue, _ = SimODE(Lib, Ξtrue, u0, tspan, dt)
        Xaic, _ = SimODE(Lib, Ξaic, u0, tspan, dt)
        Xslic, _ = SimODE(Lib, Ξslic, u0, tspan, dt)
        Xtrues = push!(Xtrues, Xtrue)
        Xaics = push!(Xaics, Xaic)
        Xslics = push!(Xslics, Xslic)
    end

    return Xtrues, Xaics, Xslics
end

u0s = hcat(Xlor[:,1], Xlor[:,1]*10, ceil.(Xlor[:,1]/10))
Ξs = [lvdata["Ξtrue"][:,:], Ξes_aic_avg, Ξes_slic_avg]
Xtrues, Xaics, Xslics = SimNewICs(LVLib, Ξs, u0s, tspan, dt)


# Make plots: 
# convenience function for plotting 
function MakeGenPlots(Xtrues, Xslics, Xaics, u0s, ind; 
    axlims_true = (0,60) ,
    axlims_slic = (0,60) ,
    axlims_aic = (0,60) ,
    lwt = 7 ,# linewidth of true data
    lwm = 3 ,# linewidth of slic/aic
    ms = 10 ,# size of marker for initial cond
    )
    p1 = plot(Xtrues[ind][1,:], Xtrues[ind][2,:], color=:black, label="True", lw=lwt, aspect_ratio=1)
    scatter!([u0s[:,ind][1]], [u0s[:,ind][2]], color=:red, ms=ms, label="IC", alpha=1) 
    #plot!(legend=false)
    xlims!(axlims_true...)
    ylims!(axlims_true...)
    xlabel!("x")
    ylabel!("y")

    p2 = plot(Xtrues[ind][1,:], Xtrues[ind][2,:], color=:black, label="True", lw=lwt, aspect_ratio=1)
    plot!(Xslics[ind][1,:], Xslics[ind][2,:], color=:cyan, label="SLIC", lw=lwm)
    scatter!([u0s[:,ind][1]], [u0s[:,ind][2]], color=:red, ms=ms, label="IC", alpha=1) 
    xlims!(axlims_slic...)
    ylims!(axlims_slic...)
    xlabel!("x")
    ylabel!("y")

    p3 = plot(Xtrues[ind][1,:], Xtrues[ind][2,:], color=:black, label="True", lw=lwt, aspect_ratio=1)
    plot!(Xaics[ind][1,:], Xaics[ind][2,:], color=:magenta, label="AICc", lw=lwm)
    scatter!([u0s[:,ind][1]], [u0s[:,ind][2]], color=:red, ms=ms, label="IC", alpha=1) 
    xlims!(axlims_aic...)
    ylims!(axlims_aic...)
    xlabel!("x")
    ylabel!("y")

    p123 = plot(p1,p2,p3, layout=(1,3), size=(1200,300), margins=5*Measures.mm)

    display(p123)
end

MakeGenPlots(Xtrues, Xslics, Xaics, u0s, 1 , axlims_true = (0,60) , axlims_slic = (0,60) , axlims_aic = (0,60))
MakeGenPlots(Xtrues, Xslics, Xaics, u0s, 2 , axlims_true = (-10,170) , axlims_slic = (-10,170) , axlims_aic = (-50,250))
MakeGenPlots(Xtrues, Xslics, Xaics, u0s, 3 , axlims_true = (-10,140) , axlims_slic = (-10,140) , axlims_aic = (-10,140))
