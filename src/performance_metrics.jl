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
    TP = mean(TP, dims=1)#/count(abs.(Ξtrue) .> 0)
    TN = mean(TN, dims=1)#/count(abs.(Ξtrue) .== 0)
    FP = mean(FP, dims=1)#/count(abs.(Ξtrue) .== 0)
    FN = mean(FN, dims=1)#/count(abs.(Ξtrue) .> 0)

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

function PerformancePlotForOneSys(Ξtrues, Ξs, label::String;
        ic_labels = ["SLIC", "AIC", "AICc", "BIC", "HQIC", "KIC", "BC"],
        colors = [:red, :black, :blue, :green, :darkorange, :gold, :purple],
        markershapes= [:circle , :utriangle, :dtriangle, :square, :star, :hexagon, :pentagon] ,
        noise_lvls = [0,5,10,20,30,40],
        lw = 3, 
        ms = 5,
        alpha= 0.7,
        fpr_ylims = 1.35*ones(5), 
        margin = 5, 
        S=3,
        legend_on = false 
    )
    ps_acc = []
    ps_fpr = []
    ps_err = []

    #S = 1:3
    for sys in 1:S
        Ξsys = Ξs[1][:,:,:,:,sys]
        Ξtrue = Ξtrues[sys] 
        cf , err = PerformanceMetrics(Ξtrue, Ξsys)
        acc = AccMetric(cf)
        fpr = FPRMetric(cf)

        # make error plot 
        err_avg, err_std = mean(err, dims=1), std(err, dims=1)

        # err plot
        psys_err = plot(noise_lvls, err_avg', lw=lw, color=colors[1], label=false, yscale=:log10, dpi=300, alpha=alpha)
        scatter!(noise_lvls, err_avg', yerror=err_std',alpha=alpha, ms=ms, markershape=markershapes[1], color=colors[1], label="$(ic_labels[1]): "*label)
        ylims!(1e-5, 1e1)
        legend = legend_on ? :bottomright : nothing
        plot!(legend=legend)

        # acc plot
        psys_acc = plot(noise_lvls, acc', lw=lw, color=colors[1], label=false, dpi=300, alpha=alpha,)
        scatter!(noise_lvls, acc',alpha=alpha,  ms=ms, color=colors[1], markershape=markershapes[1],  label="$(ic_labels[1]): "*label)
        legend = legend_on ? :bottomright : nothing
        plot!(legend=legend, yticks= [0., 0.5, 1.], yminorticks=2)
        ylims!(-0.1, 1.1)

        # fpr plot 
        psys_fpr = plot(noise_lvls, fpr',alpha=alpha, lw=lw, color=colors[1], label=false, dpi=300)
        scatter!(noise_lvls, fpr', ms=ms,alpha=alpha, color=colors[1], markershape=markershapes[1],  label="$(ic_labels[1]): "*label)
        legend = legend_on ? :topleft : nothing
        plot!(legend=legend,  yticks= [0., 0.5, 1.], yminorticks=2)
        ylims!(-0.1, fpr_ylims[sys])

        for ic=2:length(ic_labels)
            Ξsys = Ξs[ic][:,:,:,:,sys]

            cf , err = PerformanceMetrics(Ξtrue, Ξsys)
            acc = AccMetric(cf)
            fpr = FPRMetric(cf)

            # make error plot 
            err_avg, err_std = mean(err, dims=1), std(err, dims=1)

            # err plot
            plot!(psys_err, noise_lvls, err_avg', lw=lw,alpha=alpha, color=colors[ic], label=false, yscale=:log10, dpi=300)
            scatter!(psys_err, noise_lvls, err_avg',alpha=alpha, yerror=err_std', ms=ms, markershape=markershapes[ic], color=colors[ic], label="$(ic_labels[ic]): "*label)
            
            # acc plot
            psys_acc = plot!(psys_acc, noise_lvls, acc', alpha=alpha,lw=lw, color=colors[ic], label=false, dpi=300)
            scatter!(psys_acc, noise_lvls, acc', alpha=alpha, ms=ms, color=colors[ic], markershape=markershapes[ic], label="$(ic_labels[ic]): "*label)

            # fpr plot 
            psys_fpr = plot!(psys_fpr,noise_lvls, fpr',alpha=alpha, lw=lw, color=colors[ic], label=false, dpi=300)
            scatter!(psys_fpr, noise_lvls, fpr',alpha=alpha, ms=ms, color=colors[ic], markershape=markershapes[ic], label="$(ic_labels[ic]): "*label)
            
        end
        push!(ps_err, psys_err)
        push!(ps_acc, psys_acc)
        push!(ps_fpr, psys_fpr)
    end

    # this plots it as a column
    p_err = plot(ps_err... , layout=(S,1), size=(300, 300*S), dpi=300, margins=margin*Measures.mm)
    p_acc = plot(ps_acc... , layout=(S,1), size=(300, 300*S), dpi=300, margins=margin*Measures.mm)
    p_fpr = plot(ps_fpr... , layout=(S,1), size=(300, 300*S), dpi=300, margins=margin*Measures.mm)

    return p_err, p_acc, p_fpr
end

function PerformancePlots(Ξtrue_all , Ξs;
    ic_labels = ["SLIC", "AIC", "AICc", "BIC", "HQIC", "KIC", "BC"],
    labels = ["Lorenz" , "Rossler", "L-V" , "Brus.", "VdP" , "NLP"],
    colors = [:red, :black, :blue, :green, :darkorange, :gold, :purple],
    markershapes= [:circle , :utriangle, :dtriangle, :square, :star, :hexagon, :pentagon] ,
    use_std = false,
    noise_lvls = [0,5,10,20,30,40],
    lw = 3, 
    ms = 5,
    alpha= 1.,
    legend_on = false,
    fpr_ylims = 1.35*ones(6),
    err_ylims = vcat(fill((1e-4,1e0), 5), [(1e-4,1e2)])
    )
    ps_acc = []
    ps_fpr = []
    ps_err = []

    for sys=1:length(labels)

        Ξsys = Ξs[1][sys]
        Ξtrue = Ξtrue_all[sys]

        cf , err = PerformanceMetrics(Ξtrue, Ξsys)
        acc = AccMetric(cf)
        fpr = FPRMetric(cf)

        # make error plot 
        err_avg, err_std = mean(err, dims=1), std(err, dims=1)

        # err plot
        psys_err = plot(noise_lvls, err_avg', lw=lw, color=colors[1], label=false, yscale=:log10, dpi=300, alpha=alpha)
        scatter!(noise_lvls, err_avg', yerror=err_std',alpha=alpha, ms=ms, markershape=markershapes[1], color=colors[1], label="$(ic_labels[1]): "*labels[sys])
        ylims!(err_ylims[sys])
        legend = legend_on ? :bottomright : nothing
        plot!(legend=legend)

        # acc plot
        psys_acc = plot(noise_lvls, acc', lw=lw, color=colors[1], label=false, dpi=300, alpha=alpha,)
        scatter!(noise_lvls, acc',alpha=alpha,  ms=ms, color=colors[1], markershape=markershapes[1],  label="$(ic_labels[1]): "*labels[sys])
        #hline!([count(abs.(Ξtrue) .> 0)/prod(size(Ξtrue))], lw=lw, ls=:dash, label=nothing, color=:black)
        legend = legend_on ? :bottomright : nothing
        plot!(legend=legend, yticks= [0., 0.5, 1.], yminorticks=2)
        ylims!(-0.1, 1.1)

        # fpr plot 
        psys_fpr = plot(noise_lvls, fpr',alpha=alpha, lw=lw, color=colors[1], label=false, dpi=300)
        scatter!(noise_lvls, fpr', ms=ms,alpha=alpha, color=colors[1], markershape=markershapes[1],  label="$(ic_labels[1]): "*labels[sys])
        legend = legend_on ? :topleft : nothing
        plot!(legend=legend,  yticks= [0., 0.5, 1.], yminorticks=2)
        ylims!(-0.1, fpr_ylims[sys])

        for ic=2:length(ic_labels)
            Ξsys = Ξs[ic][sys]
            Ξtrue = Ξtrue_all[sys]

            cf , err = PerformanceMetrics(Ξtrue, Ξsys)
            acc = AccMetric(cf)
            fpr = FPRMetric(cf)

            # make error plot 
            err_avg, err_std = mean(err, dims=1), std(err, dims=1)

            # err plot
            plot!(psys_err, noise_lvls, err_avg', lw=lw,alpha=alpha, color=colors[ic], label=false, yscale=:log10, dpi=300)
            scatter!(psys_err, noise_lvls, err_avg',alpha=alpha, yerror=err_std', ms=ms, markershape=markershapes[ic], color=colors[ic], label="$(ic_labels[ic]): "*labels[sys])
            ylims!(err_ylims[sys])

            # acc plot
            psys_acc = plot!(psys_acc, noise_lvls, acc', alpha=alpha,lw=lw, color=colors[ic], label=false, dpi=300)
            scatter!(psys_acc, noise_lvls, acc', alpha=alpha, ms=ms, color=colors[ic], markershape=markershapes[ic], label="$(ic_labels[ic]): "*labels[sys])

            # fpr plot 
            psys_fpr = plot!(psys_fpr,noise_lvls, fpr',alpha=alpha, lw=lw, color=colors[ic], label=false, dpi=300)
            scatter!(psys_fpr, noise_lvls, fpr',alpha=alpha, ms=ms, color=colors[ic], markershape=markershapes[ic], label="$(ic_labels[ic]): "*labels[sys])
            
        end


        push!(ps_err, psys_err)
        push!(ps_acc, psys_acc)
        push!(ps_fpr, psys_fpr)
    end

    p_err = plot(ps_err... , layout=(2,3), size=(1200,600), dpi=300)
    p_acc = plot(ps_acc... , layout=(2,3), size=(1200,600), dpi=300)
    p_fpr = plot(ps_fpr... , layout=(2,3), size=(1200,600), dpi=300)

    if !legend_on
        p_leg = scatter()
        for i=1:length(ic_labels)
            scatter!([1],[2], label=ic_labels[i], color=colors[i], markershape=markershapes[i], alpha=alpha)
        end
        return p_err, p_acc, p_fpr, p_leg 
    else
        return p_err, p_acc, p_fpr
    end

end
