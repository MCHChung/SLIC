using DrWatson
@quickactivate "SLIC"
using JLD, Random

# This script generates the data used for model discovery in the SI. The data for the main text is also included. 

# functions to extract models and visualize results
include(srcdir("generate_ode_data.jl"))

# set seed 
rng = Random.default_rng()
Random.seed!(rng, 0)

function GenSIData(sys::Int, u0s, ps, tspans, Ξtrues_ps)

    @assert (1 <= sys <= 6) && length(u0s)==3 && length(ps)==2 && length(tspans)==5

    data = begin
        if sys == 1
            load(datadir("sims", "ode_data", "lordata.jld"))
        elseif sys == 2
            load(datadir("sims", "ode_data", "rossdata.jld"))
        elseif sys == 3
            load(datadir("sims", "ode_data", "lvdata.jld"))
        elseif sys == 4
            load(datadir("sims", "ode_data", "brusdata.jld"))
        elseif sys == 5
            load(datadir("sims", "ode_data", "vdpdata.jld"))
        elseif sys == 6
            load(datadir("sims", "ode_data", "nlpdata.jld"))
        end
    end
    ts = data["ts"]
    tspan = (ts[1] , ts[end])
    dt = ts[2] - ts[1]
    # generate data 
    Xtrues = []
    # parameter changes
    Xtrues_p1s = []
    Xtrues_p2s = []
    
    # sample freq changes 
    Xtrues_dt5 = []
    Xtrues_dt2 = []
    Xtrues_dtpt5 = []
    Xtrues_dtpt2 = []
    ts_dt = [0.:5*dt:ts[end] , 0.:2*dt:ts[end], 0.:0.5*dt:ts[end], 0.:0.2*dt:ts[end]]

    # traj length changes 
    Xtrues_T1 = []
    Xtrues_T2 = []
    Xtrues_T3 = []
    Xtrues_T4 = []
    Xtrues_T5 = []
    ts_T = [0:dt:tspans[1][end], 0:dt:tspans[2][end], 0:dt:tspans[3][end], 0:dt:tspans[4][end], 0:dt:tspans[5][end]]

    for i=1:3
        # in main text  
        u0 = u0s[i]
        Xtrue , _ = GenData(sys, tspan, dt , u0)
        push!(Xtrues, Xtrue)

        # diff params 
        Xtrue_p1, _ = GenData(sys, tspan, dt, u0, ps[1])
        Xtrue_p2, _ = GenData(sys, tspan, dt, u0, ps[2])
        push!(Xtrues_p1s, Xtrue_p1)
        push!(Xtrues_p2s, Xtrue_p2)

        # diff samp freq
        Xtrue_dt5, _ = GenData(sys, tspan, 5*dt, u0)
        Xtrue_dt2, _ = GenData(sys, tspan, 2*dt, u0)
        Xtrue_dtpt5, _ = GenData(sys, tspan, 0.5*dt, u0)
        Xtrue_dtpt2, _ = GenData(sys, tspan, 0.2*dt, u0)
        push!(Xtrues_dt5, Xtrue_dt5)
        push!(Xtrues_dt2, Xtrue_dt2)
        push!(Xtrues_dtpt5, Xtrue_dtpt5)
        push!(Xtrues_dtpt2, Xtrue_dtpt2)

        # diff traj length
        Xtrue_T1, _ = GenData(sys, tspans[1], dt, u0)
        Xtrue_T2, _ = GenData(sys, tspans[2], dt, u0)
        Xtrue_T3, _ = GenData(sys, tspans[3], dt, u0)
        Xtrue_T4, _ = GenData(sys, tspans[4], dt, u0)
        Xtrue_T5, _ = GenData(sys, tspans[5], dt, u0)
        push!(Xtrues_T1, Xtrue_T1)
        push!(Xtrues_T2, Xtrue_T2)
        push!(Xtrues_T3, Xtrue_T3)
        push!(Xtrues_T4, Xtrue_T4)
        push!(Xtrues_T5, Xtrue_T5)

    end

    # aggregate 
    Xtrues_dt = [Xtrues_dt5, Xtrues_dt2, Xtrues_dtpt5, Xtrues_dtpt2]
    Xtrues_T = [Xtrues_T1, Xtrues_T2, Xtrues_T3, Xtrues_T4, Xtrues_T5]
    Xtrues_ps = [Xtrues_p1s, Xtrues_p2s]


    dict_main = Dict(
        "Ξtrue" => data["Ξtrue"], 
        "Xtrues" => Xtrues, 
        "ts" => ts
    )

    dict_dt = Dict(
        "Ξtrue" => data["Ξtrue"], 
        "Xtrues" => Xtrues_dt, 
        "ts" => ts_dt
    )

    dict_T = Dict(
        "Ξtrue" => data["Ξtrue"], 
        "Xtrues" => Xtrues_T, 
        "ts" => ts_T
    )

    dict_ps = Dict(
        "Ξtrues" => Ξtrues_ps, 
        "Xtrues" => Xtrues_ps, 
        "ts" => ts
    )

    data_si = Dict(
        "dict_main" => dict_main, 
        "dict_dt" => dict_dt, 
        "dict_T" => dict_T, 
        "dict_ps" => dict_ps
    )

    return data_si
end

function GenSIData(sys::Int, ts, u0s, ps, tspans, Ξtrues_ps)

    @assert (1 <= sys <= 6) && length(u0s)==3 && length(ps)==2 && length(tspans)==5

    data = begin
        if sys == 1
            load(datadir("sims", "ode_data", "lordata.jld"))
        elseif sys == 2
            load(datadir("sims", "ode_data", "rossdata.jld"))
        elseif sys == 3
            load(datadir("sims", "ode_data", "lvdata.jld"))
        elseif sys == 4
            load(datadir("sims", "ode_data", "brusdata.jld"))
        elseif sys == 5
            load(datadir("sims", "ode_data", "vdpdata.jld"))
        elseif sys == 6
            load(datadir("sims", "ode_data", "nlpdata.jld"))
        end
    end
    tspan = (ts[1] , ts[end])
    dt = ts[2] - ts[1]
    # generate data 
    Xtrues = []
    # parameter changes
    Xtrues_p1s = []
    Xtrues_p2s = []
    
    # sample freq changes 
    Xtrues_dt5 = []
    Xtrues_dt2 = []
    Xtrues_dtpt5 = []
    Xtrues_dtpt2 = []
    ts_dt = [0.:5*dt:ts[end] , 0.:2*dt:ts[end], 0.:0.5*dt:ts[end], 0.:0.2*dt:ts[end]]

    # traj length changes 
    Xtrues_T1 = []
    Xtrues_T2 = []
    Xtrues_T3 = []
    Xtrues_T4 = []
    Xtrues_T5 = []
    ts_T = [0:dt:tspans[1][end], 0:dt:tspans[2][end], 0:dt:tspans[3][end], 0:dt:tspans[4][end], 0:dt:tspans[5][end]]

    for i=1:3
        # in main text  
        u0 = u0s[i]
        Xtrue , _ = GenData(sys, tspan, dt , u0)
        push!(Xtrues, Xtrue)

        # diff params 
        Xtrue_p1, _ = GenData(sys, tspan, dt, u0, ps[1])
        Xtrue_p2, _ = GenData(sys, tspan, dt, u0, ps[2])
        push!(Xtrues_p1s, Xtrue_p1)
        push!(Xtrues_p2s, Xtrue_p2)

        # diff samp freq
        Xtrue_dt5, _ = GenData(sys, tspan, 5*dt, u0)
        Xtrue_dt2, _ = GenData(sys, tspan, 2*dt, u0)
        Xtrue_dtpt5, _ = GenData(sys, tspan, 0.5*dt, u0)
        Xtrue_dtpt2, _ = GenData(sys, tspan, 0.2*dt, u0)
        push!(Xtrues_dt5, Xtrue_dt5)
        push!(Xtrues_dt2, Xtrue_dt2)
        push!(Xtrues_dtpt5, Xtrue_dtpt5)
        push!(Xtrues_dtpt2, Xtrue_dtpt2)

        # diff traj length
        Xtrue_T1, _ = GenData(sys, tspans[1], dt, u0)
        Xtrue_T2, _ = GenData(sys, tspans[2], dt, u0)
        Xtrue_T3, _ = GenData(sys, tspans[3], dt, u0)
        Xtrue_T4, _ = GenData(sys, tspans[4], dt, u0)
        Xtrue_T5, _ = GenData(sys, tspans[5], dt, u0)
        push!(Xtrues_T1, Xtrue_T1)
        push!(Xtrues_T2, Xtrue_T2)
        push!(Xtrues_T3, Xtrue_T3)
        push!(Xtrues_T4, Xtrue_T4)
        push!(Xtrues_T5, Xtrue_T5)

    end

    # aggregate 
    Xtrues_dt = [Xtrues_dt5, Xtrues_dt2, Xtrues_dtpt5, Xtrues_dtpt2]
    Xtrues_T = [Xtrues_T1, Xtrues_T2, Xtrues_T3, Xtrues_T4, Xtrues_T5]
    Xtrues_ps = [Xtrues_p1s, Xtrues_p2s]


    dict_main = Dict(
        "Ξtrue" => data["Ξtrue"], 
        "Xtrues" => Xtrues, 
        "ts" => ts
    )

    dict_dt = Dict(
        "Ξtrue" => data["Ξtrue"], 
        "Xtrues" => Xtrues_dt, 
        "ts" => ts_dt
    )

    dict_T = Dict(
        "Ξtrue" => data["Ξtrue"], 
        "Xtrues" => Xtrues_T, 
        "ts" => ts_T
    )

    dict_ps = Dict(
        "Ξtrues" => Ξtrues_ps, 
        "Xtrues" => Xtrues_ps, 
        "ts" => ts
    )

    data_si = Dict(
        "dict_main" => dict_main, 
        "dict_dt" => dict_dt, 
        "dict_T" => dict_T, 
        "dict_ps" => dict_ps
    )

    return data_si
end

function GenSIData(sys::Int, u0s, tspans)

    @assert (1 <= sys <= 6) && length(u0s)==3 && length(ps)==2 && length(tspans)==5

    data = begin
        if sys == 1
            load(datadir("sims", "ode_data", "lordata.jld"))
        elseif sys == 2
            load(datadir("sims", "ode_data", "rossdata.jld"))
        elseif sys == 3
            load(datadir("sims", "ode_data", "lvdata.jld"))
        elseif sys == 4
            load(datadir("sims", "ode_data", "brusdata.jld"))
        elseif sys == 5
            load(datadir("sims", "ode_data", "vdpdata.jld"))
        elseif sys == 6
            load(datadir("sims", "ode_data", "nlpdata.jld"))
        end
    end
    ts = data["ts"]
    tspan = (ts[1] , ts[end])
    dt = ts[2] - ts[1]
    # generate data 
    Xtrues = []
    
    # sample freq changes 
    Xtrues_dt5 = []
    Xtrues_dt2 = []
    Xtrues_dtpt5 = []
    Xtrues_dtpt2 = []
    ts_dt = [0.:5*dt:ts[end] , 0.:2*dt:ts[end], 0.:0.5*dt:ts[end], 0.:0.2*dt:ts[end]]

    # traj length changes 
    Xtrues_T1 = []
    Xtrues_T2 = []
    Xtrues_T3 = []
    Xtrues_T4 = []
    Xtrues_T5 = []
    ts_T = [0:dt:tspans[1][end], 0:dt:tspans[2][end], 0:dt:tspans[3][end], 0:dt:tspans[4][end], 0:dt:tspans[5][end]]

    for i=1:3
        # in main text  
        u0 = u0s[i]
        Xtrue , _ = GenData(sys, tspan, dt , u0)
        push!(Xtrues, Xtrue)

        # diff samp freq
        Xtrue_dt5, _ = GenData(sys, tspan, 5*dt, u0)
        Xtrue_dt2, _ = GenData(sys, tspan, 2*dt, u0)
        Xtrue_dtpt5, _ = GenData(sys, tspan, 0.5*dt, u0)
        Xtrue_dtpt2, _ = GenData(sys, tspan, 0.2*dt, u0)
        push!(Xtrues_dt5, Xtrue_dt5)
        push!(Xtrues_dt2, Xtrue_dt2)
        push!(Xtrues_dtpt5, Xtrue_dtpt5)
        push!(Xtrues_dtpt2, Xtrue_dtpt2)

        # diff traj length
        Xtrue_T1, _ = GenData(sys, tspans[1], dt, u0)
        Xtrue_T2, _ = GenData(sys, tspans[2], dt, u0)
        Xtrue_T3, _ = GenData(sys, tspans[3], dt, u0)
        Xtrue_T4, _ = GenData(sys, tspans[4], dt, u0)
        Xtrue_T5, _ = GenData(sys, tspans[5], dt, u0)
        push!(Xtrues_T1, Xtrue_T1)
        push!(Xtrues_T2, Xtrue_T2)
        push!(Xtrues_T3, Xtrue_T3)
        push!(Xtrues_T4, Xtrue_T4)
        push!(Xtrues_T5, Xtrue_T5)

    end

    # aggregate 
    Xtrues_dt = [Xtrues_dt5, Xtrues_dt2, Xtrues_dtpt5, Xtrues_dtpt2]
    Xtrues_T = [Xtrues_T1, Xtrues_T2, Xtrues_T3, Xtrues_T4, Xtrues_T5]


    dict_main = Dict(
        "Ξtrue" => data["Ξtrue"], 
        "Xtrues" => Xtrues, 
        "ts" => ts
    )

    dict_dt = Dict(
        "Ξtrue" => data["Ξtrue"], 
        "Xtrues" => Xtrues_dt, 
        "ts" => ts_dt
    )

    dict_T = Dict(
        "Ξtrue" => data["Ξtrue"], 
        "Xtrues" => Xtrues_T, 
        "ts" => ts_T
    )

    data_si = Dict(
        "dict_main" => dict_main, 
        "dict_dt" => dict_dt, 
        "dict_T" => dict_T 
    )

    return data_si
end

function GenSIData(sys::Int, ts, u0s, tspans)

    @assert (1 <= sys <= 6) && length(u0s)==3 && length(ps)==2 && length(tspans)==5

    data = begin
        if sys == 1
            load(datadir("sims", "ode_data", "lordata.jld"))
        elseif sys == 2
            load(datadir("sims", "ode_data", "rossdata.jld"))
        elseif sys == 3
            load(datadir("sims", "ode_data", "lvdata.jld"))
        elseif sys == 4
            load(datadir("sims", "ode_data", "brusdata.jld"))
        elseif sys == 5
            load(datadir("sims", "ode_data", "vdpdata.jld"))
        elseif sys == 6
            load(datadir("sims", "ode_data", "nlpdata.jld"))
        end
    end
    
    tspan = (ts[1] , ts[end])
    dt = ts[2] - ts[1]
    # generate data 
    Xtrues = []
    
    # sample freq changes 
    Xtrues_dt5 = []
    Xtrues_dt2 = []
    Xtrues_dtpt5 = []
    Xtrues_dtpt2 = []
    ts_dt = [0.:5*dt:ts[end] , 0.:2*dt:ts[end], 0.:0.5*dt:ts[end], 0.:0.2*dt:ts[end]]

    # traj length changes 
    Xtrues_T1 = []
    Xtrues_T2 = []
    Xtrues_T3 = []
    Xtrues_T4 = []
    Xtrues_T5 = []
    ts_T = [0:dt:tspans[1][end], 0:dt:tspans[2][end], 0:dt:tspans[3][end], 0:dt:tspans[4][end], 0:dt:tspans[5][end]]

    for i=1:3
        # in main text  
        u0 = u0s[i]
        Xtrue , _ = GenData(sys, tspan, dt , u0)
        push!(Xtrues, Xtrue)

        # diff samp freq
        Xtrue_dt5, _ = GenData(sys, tspan, 5*dt, u0)
        Xtrue_dt2, _ = GenData(sys, tspan, 2*dt, u0)
        Xtrue_dtpt5, _ = GenData(sys, tspan, 0.5*dt, u0)
        Xtrue_dtpt2, _ = GenData(sys, tspan, 0.2*dt, u0)
        push!(Xtrues_dt5, Xtrue_dt5)
        push!(Xtrues_dt2, Xtrue_dt2)
        push!(Xtrues_dtpt5, Xtrue_dtpt5)
        push!(Xtrues_dtpt2, Xtrue_dtpt2)

        # diff traj length
        Xtrue_T1, _ = GenData(sys, tspans[1], dt, u0)
        Xtrue_T2, _ = GenData(sys, tspans[2], dt, u0)
        Xtrue_T3, _ = GenData(sys, tspans[3], dt, u0)
        Xtrue_T4, _ = GenData(sys, tspans[4], dt, u0)
        Xtrue_T5, _ = GenData(sys, tspans[5], dt, u0)
        push!(Xtrues_T1, Xtrue_T1)
        push!(Xtrues_T2, Xtrue_T2)
        push!(Xtrues_T3, Xtrue_T3)
        push!(Xtrues_T4, Xtrue_T4)
        push!(Xtrues_T5, Xtrue_T5)

    end

    # aggregate 
    Xtrues_dt = [Xtrues_dt5, Xtrues_dt2, Xtrues_dtpt5, Xtrues_dtpt2]
    Xtrues_T = [Xtrues_T1, Xtrues_T2, Xtrues_T3, Xtrues_T4, Xtrues_T5]


    dict_main = Dict(
        "Ξtrue" => data["Ξtrue"], 
        "Xtrues" => Xtrues, 
        "ts" => ts
    )

    dict_dt = Dict(
        "Ξtrue" => data["Ξtrue"], 
        "Xtrues" => Xtrues_dt, 
        "ts" => ts_dt
    )

    dict_T = Dict(
        "Ξtrue" => data["Ξtrue"], 
        "Xtrues" => Xtrues_T, 
        "ts" => ts_T
    )


    data_si = Dict(
        "dict_main" => dict_main, 
        "dict_dt" => dict_dt, 
        "dict_T" => dict_T 
    )

    return data_si
end

# Lorenz
sys = 1 
u0s = [rand(rng, -50:50 , 3) for i=1:3]
tspans = [(0., 5.), (0., 6.), (0., 7.), (0., 8.), (0., 9.)]
ps = [[10., 40., 8/3] , [10., 80., 8/3]]
Ξtrues_ps = []
Ξtrue_ps1 = [-10. 10. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.; 40. -1. 0. 0. 0. -1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.; 0. 0. -8/3 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]'
Ξtrue_ps2 = [-10. 10. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.; 80. -1. 0. 0. 0. -1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.; 0. 0. -8/3 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]'
push!(Ξtrues_ps, Ξtrue_ps1)
push!(Ξtrues_ps, Ξtrue_ps2)
lordata_si = GenSIData(sys, u0s, ps, tspans, Ξtrues_ps)
#wsave(datadir("sims", "ode_data", "lordata_si.jld"), lordata_si)


# Rossler
sys = 2 
u0s = [rand(rng, 0:5 , 3) for i=1:3]
tspans = [(0., 5.), (0., 6.), (0., 7.), (0., 8.), (0., 9.)]
ts = 0:1e-3:10.
rossdata_si = GenSIData(sys, ts, u0s, tspans)
#wsave(datadir("sims", "ode_data", "rossdata_si.jld"), rossdata_si)


# L-V
sys = 3
u0s = [rand(rng, 1:50 , 2) for i=1:3]
tspans = [(0., 5.), (0., 10.), (0., 15.), (0., 20.), (0., 25.)]
ps = [[0.1], [0.5]]
Ξtrues_ps = []
Ξtrue_ps1 = [1. 0. 0. -0.1 0. 0. 0. 0. 0.; 0. -1 0. 0.1 0. 0. 0. 0. 0.]'
Ξtrue_ps2 = [1. 0. 0. -0.5 0. 0. 0. 0. 0.; 0. -1 0. 0.5 0. 0. 0. 0. 0.]'
lvdata_si = GenSIData(sys, u0s, ps, tspans, Ξtrues_ps)
push!(Ξtrues_ps, Ξtrue_ps1)
push!(Ξtrues_ps, Ξtrue_ps2)
#wsave(datadir("sims", "ode_data", "lvdata_si.jld"), lvdata_si)


# Brusselator
sys = 4
u0s = [[0. , rand(rng, 0.:10.)] for i=1:3]
tspans = [(0., 5.), (0., 6.), (0., 7.), (0., 8.), (0., 9.)]
ps = [[-3., 1], [-5., 1]]
Ξtrues_ps = []
Ξtrue_ps1 = [1. ps[1][1] 0. 0. 0. 0. 0. 1. 0. 0. ; 0. -ps[1][1]-1 0. 0. 0. 0. 0. -1. 0. 0.]'
Ξtrue_ps2 = [1. ps[2][1] 0. 0. 0. 0. 0. 1. 0. 0. ; 0. -ps[2][1]-1 0. 0. 0. 0. 0. -1. 0. 0.]'
ts = 0.:1e-3:10.
brusdata_si = GenSIData(sys, ts, u0s, ps, tspans, Ξtrues_ps)
push!(Ξtrues_ps, Ξtrue_ps1)
push!(Ξtrues_ps, Ξtrue_ps2)
#wsave(datadir("sims", "ode_data", "brusdata_si.jld"), brusdata_si)


# VdP
sys = 5
u0s = [rand(rng, 0:0.5:3 , 2) for i=1:3]
tspans = [(0., 5.), (0., 10.), (0., 15.), (0., 20.), (0., 25.)]
ps = [[0.4], [2.]]
Ξtrues_ps = []
Ξtrue_ps1 = [0. 1. 0. 0. 0. 0. 0. 0. 0.; -1 ps[1][1] 0. 0. 0. 0. -ps[1][1] 0. 0.]'
Ξtrue_ps2 = [0. 1. 0. 0. 0. 0. 0. 0. 0.; -1 ps[2][1] 0. 0. 0. 0. -ps[2][1] 0. 0.]'
vdpdata_si = GenSIData(sys, u0s, ps,tspans, Ξtrues_ps, )
push!(Ξtrues_ps, Ξtrue_ps1)
push!(Ξtrues_ps, Ξtrue_ps2)
#wsave(datadir("sims", "ode_data", "vdpdata_si.jld"), vdpdata_si)


# NLP
sys = 6
u0s = [[rand(rng, π/2:0.1:3π/4), 0.] for i=1:3]
tspans = [(0., 5.), (0., 10.), (0., 15.), (0., 20.), (0., 25.)]
nlpdata_si = GenSIData(sys, u0s, tspans)
#wsave(datadir("sims", "ode_data", "nlpdata_si.jld"), nlpdata_si)
