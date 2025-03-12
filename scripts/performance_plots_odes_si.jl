using DrWatson
@quickactivate "SLIC"

# This file reproduces the performance of SLIC on ODEs across changes in trajectory length, sampling frequency, parameters changes, and initial conditions


# get performance metrics 
include(srcdir("performance_metrics.jl"))

# plot settings
default(dpi=300, fontfamily="computer modern", grid=false)


# INPUTS: 
# sys = ode system : 1=lorenz , 2=ross, 3=lv, 4=brus, 5=vdp, 6=nlp
# i = SI data :  1=init conds, 2=params, 3=samp freq, 4=traj_length

# OUTPUTS:
# p_err = plot of scaled L1 error for given condition & system
# p_acc = plot of accuracy in recovering model coefficients for given condition & system
# p_fpr = plot of false-positive rate in recovering model coefficients for given condition & system

# NOTE:
# for each of these plots, the x-axis is the noise level and the y-axis is the metric 
# each plot in a column is a different condition tested for the 'i' specified (e.g. if i=1, then each plot in the column is a separate IC)
function MakeSIODEPlots(sys::Int, i::Int; legend_on = false)
    @assert 1<=sys<=6 && 1<=i<=6
    #=
    if i==2 && (sys == 2 || sys == 6) 
        throw("If i=2, sys=1,3,4,5")
    end 
    =# 
    sys_str = begin
        if sys==1 
            "lor"
        elseif sys==2 
            "ross"
        elseif sys==3 
            "lv"
        elseif sys==4 
            "brus"
        elseif sys==5 
            "vdp"
        elseif sys==6 
            "nlp"
        end
    end

    cond_str , S = begin
        if i==1 
            "ss" , 3
        elseif i==2 
            "Ts" , 5
        elseif i==3 
            "dts" , 4
        elseif i==4 
            "ps" , 2
        end
    end

    file_str = sys_str*"_results_si.jld2"

    data = wload(datadir("sims", "ode_results_si", file_str))["results_"*cond_str] 
    labels = ["Ξslics", "Ξaics", "Ξaiccs", "Ξbics", "Ξhqics", "Ξkics", "Ξbcs"]
    
   
    Ξtrues,  Ξs = [], [] 

    if sys == 5 && i != 4   
        # create vec of true library terms 
        Ξtrues = []
        Ξtrue_vdp_dp = similar(data["Ξtrue"]) 
        # to get accurate metrics for VdP, we duplicate the 2nd column. 1st column (dx/dt = v) is a trivial prediction
        Ξtrue_vdp_dp[:,1] = Ξtrue_vdp_dp[:,2] = data["Ξtrue"][:,2]
        Ξtrue = Ξtrue_vdp_dp ;
        for i=1:S
            push!(Ξtrues, Ξtrue)
        end

        Ξs = []
        for ic=1:length(labels)
            #Ξs_vdp_dp = similar(data[labels[ic]][:,:,:,:,:1])
            Ξs_vdp_dp = similar(data[labels[ic]])
            Ξs_vdp_dp[:, 1, :, :, :] = Ξs_vdp_dp[:, 2, :, :, :] = data[labels[ic]][:,2,:,:,:]
            push!(Ξs, Ξs_vdp_dp)
        end
        Ξtrue, Ξs
    elseif sys == 5 && i == 4
        # creating vec of true library terms 
        Ξtrues = []
        for i=1:S
            Ξtrue_vdp_dp = similar(data["Ξtrue"][i])
            # to get accurate metrics for VdP, we duplicate the 2nd column. 1st column (dx/dt = v) is a trivial prediction
            Ξtrue_vdp_dp[:,1] = Ξtrue_vdp_dp[:,2] = data["Ξtrue"][i][:,2]
            push!(Ξtrues, Ξtrue_vdp_dp)
        end

        Ξs = []
        for ic=1:length(labels)
            Ξs_vdp_dp = similar(data[labels[ic]][:,:,:,:,:])
            Ξs_vdp_dp[:, 1, :, :, :] = Ξs_vdp_dp[:, 2, :, :, :] = data[labels[ic]][:,2,:,:,:]
            push!(Ξs, Ξs_vdp_dp)
        end
        Ξtrues, Ξs
    elseif sys !=5 &&  i != 4
        # create vec of true library terms
        Ξtrues = []
        for i=1:S
            push!(Ξtrues, data["Ξtrue"])
        end

        Ξs = []
        for ic=1:length(labels)
            push!(Ξs, data[labels[ic]])
        end
        Ξtrues, Ξs
    elseif sys != 5 && i == 4
        key = "Ξtrues" ∈ keys(data) ? "Ξtrues" : "Ξtrue"
        Ξtrues = data[key]
        Ξs = []
        for ic=1:length(labels)
            push!(Ξs, data[labels[ic]])
        end
        Ξtrues , Ξs
    end
    
    
    p_err, p_acc, p_fpr = PerformancePlotForOneSys(Ξtrues, Ξs, sys_str, noise_lvls = [0,5,10,20,30,40], S=S, 
    margin=7, legend_on=legend_on)
    
    return p_err, p_acc, p_fpr
end


sys_labels = ["lor" , "ross", "lv", "brus", "vdp", "nlp"]
si_labels = ["ss", "Ts", "dts", "ps"]


for i=1:length(si_labels)
    
    for sys=1:length(sys_labels)
        println("i = $(i) , sys = $(sys)")
        if i == 4 && (sys == 6 || sys == 2)
            continue
        end
        p_err, p_acc, p_fpr = MakeSIODEPlots(sys, i, legend_on=false)
        sys_str = sys_labels[sys]
        si_str = si_labels[i]
        #=
        # save plots
        wsave(plotsdir(sys_str*"_acc_"*si_str*".png"), p_acc)
        wsave(plotsdir(sys_str*"_fpr_"*si_str*".png"), p_fpr)
        wsave(plotsdir(sys_str*"_err_"*si_str*".png"), p_err)
        =# 
    end
end
