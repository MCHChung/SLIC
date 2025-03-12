using DrWatson
@quickactivate "SLIC"

using Plots, JLD 

# This script just makes plots for the ODE systems that had varied parameters 

# plot settings
default(dpi=300, fontfamily="computer modern", grid=false, framestyle=:default)

# load data  
lordata = load(datadir("sims", "ode_data", "lordata_si.jld"))["dict_ps"]
lvdata = load(datadir("sims", "ode_data", "lvdata_si.jld"))["dict_ps"]
brusdata = load(datadir("sims", "ode_data", "brusdata_si.jld"))["dict_ps"]
vdpdata = load(datadir("sims", "ode_data", "vdpdata_si.jld"))["dict_ps"]

# parameters for plots 
color= :black

# make plots
# ============= Lorenz =============
# cond 1
Xs1 = lordata["Xtrues"][1][1]
plor_1 = plot(Xs1[1,:], Xs1[2,:], Xs1[3,:],lw=3, color=color, label=false)
xlabel!("x")
ylabel!("y")
zlabel!("z")

Xs2 = lordata["Xtrues"][2][1]
plor_2 = plot(Xs2[1,:], Xs2[2,:], Xs2[3,:],lw=3, color=color, label=false)
xlabel!("x")
ylabel!("y")
zlabel!("z")

plor_12 = plot(plor_1, plor_2, layout=(2,1), size=(400, 800))
#wsave(plotsdir("si", "lor_param.png"), plor_12)

display(plor_12)

# ============= L-V =============
Xs1 = lvdata["Xtrues"][1][1]
plv_1 = plot(Xs1[1,:], Xs1[2,:], lw=3, color=color, label=false, size=(400,400))
xlabel!("x")
ylabel!("y")
zlabel!("z")

Xs2 = lvdata["Xtrues"][2][1]
plv_2 = plot(Xs2[1,:], Xs2[2,:],lw=3, color=color, label=false, size=(400,400))
xlabel!("x")
ylabel!("y")
zlabel!("z")

plv_12 = plot(plv_1, plv_2, layout=(2,1), size=(400, 800))
#wsave(plotsdir("si", "lv_param.png"), plv_12)

display(plv_12)

# ============= Brusselator =============
Xs1 = brusdata["Xtrues"][1][1]
pbrus_1 = plot(Xs1[1,:], Xs1[2,:], lw=3, color=color, label=false, size=(400,400))
xlabel!("x")
ylabel!("y")
zlabel!("z")

Xs2 = brusdata["Xtrues"][2][1]
pbrus_2 = plot(Xs2[1,:], Xs2[2,:], lw=3, color=color, label=false, size=(400,400))
xlabel!("x")
ylabel!("y")
zlabel!("z")

pbrus_12 = plot(pbrus_1, pbrus_2, layout=(2,1), size=(400, 800))
#wsave(plotsdir("si", "brus_param.png"), pbrus_12)

display(pbrus_12)

# ============= VdP =============
Xs1 = vdpdata["Xtrues"][1][1]
pvdp_1 = plot(Xs1[1,:], Xs1[2,:], lw=3, color=color, label=false,size=(400,400))
xlabel!("x")
ylabel!("y")
zlabel!("z")

Xs2 = vdpdata["Xtrues"][2][1]
pvdp_2 = plot(Xs2[1,:], Xs2[2,:], lw=3, color=color, label=false, size=(400,400))
xlabel!("x")
ylabel!("y")
zlabel!("z")

pvdp_12 = plot(pvdp_1, pvdp_2, layout=(2,1), size=(400, 800))
#wsave(plotsdir("si", "vdp_param.png"), pvdp_12)

display(pvdp_12)
