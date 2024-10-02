using Plots, StatsBase, LinearAlgebra, Measures

# INPUTS: 
# Ξtrue => true coeff matrix 
# Ξs => matrix of coeffs from algo , shape = (# lib funcs, # state vars, # runs at fixed noise lvl, # noise lvls)

# OUTPUTS: 
# (TP, TN, FP, FN) => the confusion matrix, essentially. TP = true +, TN = true -, FP = false + , FN = false - 
#  err = model errors 

function PerformanceMetrics(Ξtrue, Ξs)
    # Confusion matrix 
    TP = zeros(size(Ξs)[3:4]) # true pos rate 
    TN = zeros(size(Ξs)[3:4]) # true neg rate 
    FP = zeros(size(Ξs)[3:4]) # false pos rate 
    FN = zeros(size(Ξs)[3:4]) # false neg rate

    # compute the confusion matrix quantities 
    Ξtrue_bin = abs.(Ξtrue) .> 0.
    Ξs_bin = abs.(Ξs) .> 0.
    for n_lvl=1:size(Ξs,4)
        for n_run=1:size(Ξs,3)
             for i=1:size(Ξs,1), j=1:size(Ξs,2)
                c_true = Ξtrue_bin[i,j]
                c_pred = Ξs_bin[i,j,n_run,n_lvl]
                if c_pred == c_true && c_pred == 1
                    TP[n_run, n_lvl] += 1 
                    continue
                elseif c_pred == c_true && c_pred == 0
                    TN[n_run, n_lvl] += 1
                    continue
                elseif c_pred != c_true && c_pred == 1
                    FP[n_run, n_lvl] += 1
                    continue
                elseif c_pred != c_true && c_pred == 0
                    FN[n_run, n_lvl] += 1
                    continue
                end
             end
        end
    end
    
    # average for each noise level 
    TP = 100*mean(TP, dims=1)/count(abs.(Ξtrue) .> 0)
    TN = 100*mean(TN, dims=1)/count(abs.(Ξtrue) .== 0)
    FP = 100*mean(FP, dims=1)/count(abs.(Ξtrue) .== 0)
    FN = 100*mean(FN, dims=1)/count(abs.(Ξtrue) .> 0)

    # compute average relative l1 error for coeffs 
    err = (sum(abs, Ξs .- Ξtrue, dims=(1,2))/sum(abs, Ξtrue))[1,1,:,:]

    return (TP, TN, FP, FN) , err
end

# the following 2 functions simply compute the Accuracy and False positive rate (FPR) from confusion matrix (cf)
function AccMetric(cf)
    # (TP + TN)/(TP + TN + FP + FN)
    return @. (cf[1] + cf[2])/(cf[1]+cf[2]+cf[3]+cf[4])
end

function FPRMetric(cf)
    # (FP)/(TN + FP)
    return @. (cf[3])/(cf[2]+cf[3])
end

# The following functions make plots from metrics 

function PerformancePlots(Ξtrue_all , Ξs_all_slic, Ξs_all_aic;
        labels = ["Lorenz" , "Rossler", "L-V" , "Brus.", "VdP" , "NLP"],
        use_std = false,
        noise_lvls = [0,5,10,20,30,40],
        lw = 3, 
        ms = 5,
        fpr_ylims = 1.35*ones(6)
    )
    ps_acc = []
    ps_fpr = []
    ps_err = []
    
    for sys=1:length(labels)
        Ξsys_slic = Ξs_all_slic[sys]
        Ξsys_aic = Ξs_all_aic[sys]
        Ξtrue = Ξtrue_all[sys]

        cf_slic , err_slic = PerformanceMetrics(Ξtrue, Ξsys_slic)
        cf_aic ,  err_aic = PerformanceMetrics(Ξtrue, Ξsys_aic)

        # make error plot 
        err_avg_slic, err_std_slic = mean(err_slic, dims=1), std(err_slic, dims=1)
        err_avg_aic, err_std_aic = mean(err_aic, dims=1), std(err_aic, dims=1)

        psys_err = plot(noise_lvls, err_avg_slic', lw=lw, color=:red, label=false, yscale=:log10, dpi=300)
        scatter!(noise_lvls, err_avg_slic', yerror=err_std_slic', ms=ms, color=:red, label="SLIC: "*labels[sys])

        plot!(noise_lvls, err_avg_aic', lw=lw, color=:black, label=false)
        scatter!(noise_lvls, err_avg_aic', yerror=err_std_aic', markershape=:utriangle,  ms=ms, color=:black, label="AICc: "*labels[sys])

        ylims!(1e-5, 1e1)
        plot!(legend=:bottomright)
        push!(ps_err, psys_err)

        # make acc plots 
        acc_slic = AccMetric(cf_slic)
        acc_aic = AccMetric(cf_aic)

        psys_acc = plot(noise_lvls, acc_slic', lw=lw, color=:red, label=false, dpi=300)
        scatter!(noise_lvls, acc_slic',  ms=ms, color=:red, label="SLIC: "*labels[sys])
        plot!(noise_lvls, acc_aic', lw=lw, color=:black, label=false)
        scatter!(noise_lvls, acc_aic', ms=ms, color=:black, markershape=:utriangle,  label="AICc: "*labels[sys])
        plot!(legend=:bottomright, yticks= [0., 0.5, 1.], yminorticks=2)
        ylims!(-0.1, 1.1)

        push!(ps_acc, psys_acc)

        # make fpr plots 
        fpr_slic = FPRMetric(cf_slic)
        fpr_aic = FPRMetric(cf_aic)

        psys_fpr = plot(noise_lvls, fpr_slic', lw=lw, color=:red, label=false, dpi=300)
        scatter!(noise_lvls, fpr_slic', ms=ms, color=:red, label="SLIC: "*labels[sys])
        plot!(noise_lvls, fpr_aic', lw=lw, color=:black, label=false)
        scatter!(noise_lvls, fpr_aic', ms=ms, color=:black, markershape=:utriangle,  label="AICc: "*labels[sys])
        plot!(legend=:topright,  yticks= [0., 0.5, 1.], yminorticks=2)
        ylims!(-0.1, fpr_ylims[sys])

        push!(ps_fpr, psys_fpr)

    end

    p_err = plot(ps_err... , layout=(2,3), size=(1200,600), dpi=300)
    p_acc = plot(ps_acc... , layout=(2,3), size=(1200,600), dpi=300)
    p_fpr = plot(ps_fpr... , layout=(2,3), size=(1200,600), dpi=300)

    return p_err, p_acc, p_fpr
end

function PerformancePlotForOneSys(Ξtrue, Ξs_slic, Ξs_aic, label::String;
        noise_lvls = [0,5,10,20,30,40],
        lw = 3, 
        ms = 5,
        fpr_ylims = 1.35*ones(5), 
        margin = 5, 
        S=3 
    )
    ps_acc = []
    ps_fpr = []
    ps_err = []

    #S = 1:3
    for sys in 1:S
        Ξsys_pf = Ξs_slic[:,:,:,:,sys]
        Ξsys_aic = Ξs_aic[:,:,:,:,sys]

        cf_pf , err_pf = PerformanceMetrics(Ξtrue, Ξsys_pf)
        cf_aic ,  err_aic = PerformanceMetrics(Ξtrue, Ξsys_aic)

        # make error plot 
        err_avg_pf, err_std_pf = mean(err_pf, dims=1), std(err_pf, dims=1)
        err_avg_aic, err_std_aic = mean(err_aic, dims=1), std(err_aic, dims=1)

        psys_err = plot(noise_lvls, err_avg_pf', lw=lw, color=:red, label=false, yscale=:log10, dpi=300)
        #scatter!(noise_lvls, err_avg_pf', yerror=err_std_pf'/sqrt(size(err_pf,1)), ms=ms, color=sys, label="SLIC: "*labels[sys])
        scatter!(noise_lvls, err_avg_pf', yerror=err_std_pf', ms=ms, color=:red, label="SLIC: "*label, legendfontsize=14)

        plot!(noise_lvls, err_avg_aic', lw=lw, color=:black, label=false)
        #scatter!(noise_lvls, err_avg_aic', yerror=err_std_aic'/sqrt(size(err_aic,1)), markershape=:utriangle,  ms=ms, color=sys, label="AICc: "*labels[sys])
        scatter!(noise_lvls, err_avg_aic', yerror=err_std_aic', markershape=:utriangle,  ms=ms, color=:black, label="AICc: "*label, legendfontsize=14)

        ylims!(1e-5, 1e1)
        plot!(legend=:bottomright)
        push!(ps_err, psys_err)

        # make acc plots 
        acc_pf = AccMetric(cf_pf)
        acc_aic = AccMetric(cf_aic)

        psys_acc = plot(noise_lvls, acc_pf', lw=lw, color=:red, label=false, dpi=300)
        scatter!(noise_lvls, acc_pf',  ms=ms, color=:red, label="SLIC: "*label, legendfontsize=14)
        plot!(noise_lvls, acc_aic', lw=lw, color=:black, label=false)
        scatter!(noise_lvls, acc_aic', ms=ms, color=:black, markershape=:utriangle,  label="AICc: "*label, legendfontsize=14)
        plot!(legend=:bottomright, yticks= [0., 0.5, 1.], yminorticks=2)
        ylims!(-0.1, 1.1)

        push!(ps_acc, psys_acc)

        # make fpr plots 
        fpr_pf = FPRMetric(cf_pf)
        fpr_aic = FPRMetric(cf_aic)

        psys_fpr = plot(noise_lvls, fpr_pf', lw=lw, color=:red, label=false, dpi=300)
        scatter!(noise_lvls, fpr_pf', ms=ms, color=:red, label="SLIC: "*label, legendfontsize=14)
        plot!(noise_lvls, fpr_aic', lw=lw, color=:black, label=false)
        scatter!(noise_lvls, fpr_aic', ms=ms, color=:black, markershape=:utriangle,  label="AICc: "*label, legendfontsize=14)
        plot!(legend=:topright,  yticks= [0., 0.5, 1.], yminorticks=2)
        ylims!(-0.1, fpr_ylims[sys])

        push!(ps_fpr, psys_fpr)

    end

    # this plots it as a column
    p_err = plot(ps_err... , layout=(S,1), size=(300, 300*S), dpi=300, margins=margin*Measures.mm)
    p_acc = plot(ps_acc... , layout=(S,1), size=(300, 300*S), dpi=300, margins=margin*Measures.mm)
    p_fpr = plot(ps_fpr... , layout=(S,1), size=(300, 300*S), dpi=300, margins=margin*Measures.mm)

    return p_err, p_acc, p_fpr
end

function PerformancePlotForOneSys(Ξtrues::Array{Float64, 3}, Ξs_slic, Ξs_aic, label::String;
        noise_lvls = [0,5,10,20,30,40],
        lw = 3, 
        ms = 5,
        fpr_ylims = 1.35*ones(5), 
        margin = 5 ,
        S = 3
    )
    ps_acc = []
    ps_fpr = []
    ps_err = []

    for sys=1:S
        Ξtrue = Ξtrues[:,:,sys]
        Ξsys_pf = Ξs_slic[:,:,:,:,sys]
        Ξsys_aic = Ξs_aic[:,:,:,:,sys]

        cf_pf , err_pf = PerformanceMetrics(Ξtrue, Ξsys_pf)
        cf_aic ,  err_aic = PerformanceMetrics(Ξtrue, Ξsys_aic)

        # make error plot 
        err_avg_pf, err_std_pf = mean(err_pf, dims=1), std(err_pf, dims=1)
        err_avg_aic, err_std_aic = mean(err_aic, dims=1), std(err_aic, dims=1)

        psys_err = plot(noise_lvls, err_avg_pf', lw=lw, color=:red, label=false, yscale=:log10, dpi=300)
        #scatter!(noise_lvls, err_avg_pf', yerror=err_std_pf'/sqrt(size(err_pf,1)), ms=ms, color=sys, label="SLIC: "*labels[sys])
        scatter!(noise_lvls, err_avg_pf', yerror=err_std_pf', ms=ms, color=:red, label="SLIC: "*label, legendfontsize=14)

        plot!(noise_lvls, err_avg_aic', lw=lw, color=:black, label=false)
        #scatter!(noise_lvls, err_avg_aic', yerror=err_std_aic'/sqrt(size(err_aic,1)), markershape=:utriangle,  ms=ms, color=sys, label="AICc: "*labels[sys])
        scatter!(noise_lvls, err_avg_aic', yerror=err_std_aic', markershape=:utriangle,  ms=ms, color=:black, label="AICc: "*label, legendfontsize=14)

        ylims!(1e-5, 1e1)
        plot!(legend=:bottomright)
        push!(ps_err, psys_err)

        # make acc plots 
        acc_pf = AccMetric(cf_pf)
        acc_aic = AccMetric(cf_aic)

        psys_acc = plot(noise_lvls, acc_pf', lw=lw, color=:red, label=false, dpi=300)
        scatter!(noise_lvls, acc_pf',  ms=ms, color=:red, label="SLIC: "*label, legendfontsize=14)
        plot!(noise_lvls, acc_aic', lw=lw, color=:black, label=false)
        scatter!(noise_lvls, acc_aic', ms=ms, color=:black, markershape=:utriangle,  label="AICc: "*label, legendfontsize=14)
        plot!(legend=:bottomright, yticks= [0., 0.5, 1.], yminorticks=2)
        ylims!(-0.1, 1.1)

        push!(ps_acc, psys_acc)

        # make fpr plots 
        fpr_pf = FPRMetric(cf_pf)
        fpr_aic = FPRMetric(cf_aic)

        psys_fpr = plot(noise_lvls, fpr_pf', lw=lw, color=:red, label=false, dpi=300)
        scatter!(noise_lvls, fpr_pf', ms=ms, color=:red, label="SLIC: "*label, legendfontsize=14)
        plot!(noise_lvls, fpr_aic', lw=lw, color=:black, label=false)
        scatter!(noise_lvls, fpr_aic', ms=ms, color=:black, markershape=:utriangle,  label="AICc: "*label, legendfontsize=14)
        plot!(legend=:topright,  yticks= [0., 0.5, 1.], yminorticks=2)
        ylims!(-0.1, fpr_ylims[sys])

        push!(ps_fpr, psys_fpr)

    end
 
    p_err = plot(ps_err... , layout=(S,1), size=(300, 300*S), dpi=300, margins=margin*Measures.mm)
    p_acc = plot(ps_acc... , layout=(S,1), size=(300, 300*S), dpi=300, margins=margin*Measures.mm)
    p_fpr = plot(ps_fpr... , layout=(S,1), size=(300, 300*S), dpi=300, margins=margin*Measures.mm)
    return p_err, p_acc, p_fpr
end