using DrWatson
@quickactivate "SLIC"

# This file reproduces the performance of SLIC on ODEs across changes in trajectory length, sampling frequency, parameters changes, and initial conditions


# get performance metrics 
include(srcdir("performance_metrics.jl"))

# plot settings
default(dpi=300, fontfamily="computer modern", grid=false)

# get SI data 
files = readdir(datadir("sims", "ode_results_si")) 

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
function MakeSIODEPlots(sys::Int, i::Int)
    @assert 1<=sys<=6 && 1<=i<=4
    if i==2 && (sys == 2 || sys == 6) 
        throw("If i=2, sys=1,3,4,5")
    end 
    
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
            "ic" , 3
        elseif i==2 
            "param" , 2
        elseif i==3 
            "samp" , 4
        elseif i==4 
            "traj" , 5
        end
    end

    file_str = sys_str*"_"*cond_str*"_results.jld"

    data = load(datadir("sims", "ode_results_si", files[i], file_str)) 
    p_err, p_acc, p_fpr = PerformancePlotForOneSys(data["Ξtrue"], data["Ξes_slic"], data["Ξes_aic"], sys_str, 
    noise_lvls = [0,5,10,20], S=S, margin=7)
     
    return p_err, p_acc, p_fpr
end


i = 1
sys = 1 
p_err, p_acc, p_fpr = MakeSIODEPlots(sys, i)
display(p_acc)