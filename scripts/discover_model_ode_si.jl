using DrWatson
@quickactivate "SLIC"
using JLD

# functions to extract models and visualize results
include(srcdir("sparse_regress.jl"))
include(srcdir("derivative.jl"))
include(srcdir("galerkin_proj.jl"))
include(srcdir("smooth.jl"))
include(srcdir("vis_results.jl"))


# INPUTS: 
# sys = system to perform model discovery on 
# NoisePct = % noise defined as std(noise)/100*std(data)
# runs = number of times we randomly instantiate the noise at fixed noise %
# tol = prob thresholding 
# c = helps avoids log(MSE) -> infinity as MSE -> 0, (see sparse_regress.jl)
# s = subsampling index (size of jump in sliding window), s=1 indicates no subsampling 

# OUTPUTS: 
# Ξtrue = true model coefficent 
# Ξs = ic-determined models  


function DiscoverSIODEModel_ss(sys::Int, NoisePcts::Vector{Int64}, runs::Int; 
    tol=0.7, 
    c=1e-2, 
    ss=1:3, 
    num_batches=10
    )
    @assert 1 <= sys <= 6 
    if sys == 1 # Lorenz
        # get data 
        data = load(datadir("sims", "ode_data", "lordata_si.jld"))
        data = data["dict_main"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(ss))
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)
        # define lib 
        function LorLib(X, ts, wind; p=10, Δ=1)
            θ = wInt_sc(X[1,:], ts, wind, p=p,Δ=Δ)
            for i=2:size(X,1)
                θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
            end
        
            for i=1:size(X,1)
                for j=i:size(X,1)
                    θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
                end
            end
            
            for i=1:size(X,1)
                for j=i:size(X,1)
                    for k=j:size(X,1)
                        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
                    end
                end
            end
            
            return θ
        end

        for i=1:length(ss)
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs 
                    println("ss => $(ss[i]), NoisePct => $(NoisePct), run => $(run)")
                    # get inputs
                    qt, θ =  GetInputs(sys, data["Xtrues"], data["ts"] , LorLib, NoisePct, Δ=ss[i])

                    Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,i] = Ξaicc
                    Ξs_slic[:,:,run,j,i] = Ξslic
                    Ξs_bic[:,:,run,j,i] = Ξbic
                    Ξs_aic[:,:,run,j,i] = Ξaic
                    Ξs_hqic[:,:,run,j,i] = Ξhqic
                    Ξs_bc[:,:,run,j,i] = Ξbc
                    Ξs_kic[:,:,run,j,i] = Ξkic

                end
            end
        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 2 # Rossler
        # get data 
        data = load(datadir("sims", "ode_data", "rossdata_si.jld"))
        data = data["dict_main"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(ss))
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)

         # define lib 
         function RossLib(X, ts, wind; p=10, Δ=1)
            n = size(X,1)
            θ = wInt_sc(ones(size(X[1,:])), ts, wind, p=p,Δ=Δ)
            for i=1:n
                θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
            end
        
            for i=1:n
                for j=i:n
                    θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
                end
            end
            
            for i=1:n
                for j=i:n
                    for k=j:n
                        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
                    end
                end
            end
            
            return θ
        end

        # add noise
        for i=1:length(ss)
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs
                    println("ss => $(ss[i]), NoisePct => $(NoisePct), run => $(run)")

                    qt, θ = GetInputs(sys, data["Xtrues"], data["ts"], RossLib, NoisePct, Δ=ss[i])

                    Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,i] = Ξaicc
                    Ξs_slic[:,:,run,j,i] = Ξslic
                    Ξs_bic[:,:,run,j,i] = Ξbic
                    Ξs_aic[:,:,run,j,i] = Ξaic
                    Ξs_hqic[:,:,run,j,i] = Ξhqic
                    Ξs_bc[:,:,run,j,i] = Ξbc
                    Ξs_kic[:,:,run,j,i] = Ξkic

                end
            end
        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 3 # Lotka-Volterra
        # get data 
        data = load(datadir("sims", "ode_data", "lvdata_si.jld"))
        data = data["dict_main"]

        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(ss))
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)

        # define lib 
        function LVLib(X, ts, wind; p=10, Δ=1)
            n = size(X,1)
            θ = wInt_sc(X[1,:], ts, wind, p=p,Δ=Δ)
            for i=2:n
                θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
            end
        
            for i=1:n
                for j=i:n
                    θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
                end
            end
            
            for i=1:n
                for j=i:n
                    for k=j:n
                        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
                    end
                end
            end
            
            return θ
        end

        for i=1:length(ss)
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs
                    println("ss => $(ss[i]), NoisePct => $(NoisePct), run => $(run)")

                    # get inputs 
                    qt, θ = GetInputs(sys, data["Xtrues"], data["ts"], LVLib, NoisePct, Δ=ss[i])
                    
                    # determine dynamics
                    Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,i] = Ξaicc
                    Ξs_slic[:,:,run,j,i] = Ξslic
                    Ξs_bic[:,:,run,j,i] = Ξbic
                    Ξs_aic[:,:,run,j,i] = Ξaic
                    Ξs_hqic[:,:,run,j,i] = Ξhqic
                    Ξs_bc[:,:,run,j,i] = Ξbc
                    Ξs_kic[:,:,run,j,i] = Ξkic

                end
            end
        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 4 # Brusselator
        # get data 
        data = load(datadir("sims", "ode_data", "brusdata_si.jld"))
        data = data["dict_main"]

        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(ss))
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)

        # define lib 
        function BrusLib(X, ts, wind; p=10, Δ=1)
            n = size(X,1)
            θ = wInt_sc(ones(size(X[1,:])), ts, wind, p=p,Δ=Δ)
            for i=1:n
                θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
            end
        
            for i=1:n
                for j=i:n
                    θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
                end
            end
            
            for i=1:n
                for j=i:n
                    for k=j:n
                        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
                    end
                end
            end
            
            return θ
        end

        for i=1:length(ss)
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs
                    println("ss => $(ss[i]), NoisePct => $(NoisePct), run => $(run)")

                    # get inputs
                    qt, θ = GetInputs(sys, data["Xtrues"], data["ts"], BrusLib, NoisePct, Δ=ss[i])
                    
                    # discover dynamic model
                    Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,i] = Ξaicc
                    Ξs_slic[:,:,run,j,i] = Ξslic
                    Ξs_bic[:,:,run,j,i] = Ξbic
                    Ξs_aic[:,:,run,j,i] = Ξaic
                    Ξs_hqic[:,:,run,j,i] = Ξhqic
                    Ξs_bc[:,:,run,j,i] = Ξbc
                    Ξs_kic[:,:,run,j,i] = Ξkic

                end
            end
        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 5 # Van der Pol
        # get data 
        data = load(datadir("sims", "ode_data", "vdpdata_si.jld"))
        data = data["dict_main"]

        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(ss))
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)
        
        # define lib 
        function VdPLib(X, ts, wind; p=10, Δ=1)
            n = size(X,1)
            θ = wInt_sc(X[1,:], ts, wind, p=p,Δ=Δ)
            for i=2:n
                θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
            end
        
            for i=1:n
                for j=i:n
                    θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
                end
            end
            
            for i=1:n
                for j=i:n
                    for k=j:n
                        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
                    end
                end
            end
            
            return θ
        end

        for i=1:length(ss)
            for j=1:length(NoisePcts)
                NoisePct=NoisePcts[j]
                for run=1:runs
                    println("ss => $(ss[i]), NoisePct => $(NoisePct), run => $(run)")

                    # get inputs
                    qt, θ = GetInputs(sys, data["Xtrues"], data["ts"], VdPLib, NoisePct, Δ=ss[i])
                    
                    # discover model
                    Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,i] = Ξaicc
                    Ξs_slic[:,:,run,j,i] = Ξslic
                    Ξs_bic[:,:,run,j,i] = Ξbic
                    Ξs_aic[:,:,run,j,i] = Ξaic
                    Ξs_hqic[:,:,run,j,i] = Ξhqic
                    Ξs_bc[:,:,run,j,i] = Ξbc
                    Ξs_kic[:,:,run,j,i] = Ξkic

                end 
            end
        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 6 # Nonlinear Pendulum
        # get data 
        data = load(datadir("sims", "ode_data", "nlpdata_si.jld"))
        data = data["dict_main"]

        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(ss))
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)

        # define lib 
        function NLPLib(X, ts, wind; p=10, Δ=1)
            θ = wInt_sc(ones(length(X[:])), ts, wind, p=p, Δ=Δ)
            for i=1:5
                θ = hcat(θ, wInt_sc(sin.(i*X[:]), ts, wind, p=p, Δ=Δ))
                θ = hcat(θ, wInt_sc(cos.(i*X[:]), ts, wind, p=p, Δ=Δ))
            end
            return θ
        end

        for i=1:length(ss)
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs
                    println("ss => $(ss[i]), NoisePct => $(NoisePct), run => $(run)")

                    # add noise
                    qt, θ = GetInputs(sys, data["Xtrues"] , data["ts"], NLPLib, NoisePct; p=10, Δ=ss[i])

                    Ξslic, _ = EnAdSR(θ, qt, "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt, "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt, "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt, "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt, "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt, "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt, "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,i] = Ξaicc
                    Ξs_slic[:,:,run,j,i] = Ξslic
                    Ξs_bic[:,:,run,j,i] = Ξbic
                    Ξs_aic[:,:,run,j,i] = Ξaic
                    Ξs_hqic[:,:,run,j,i] = Ξhqic
                    Ξs_bc[:,:,run,j,i] = Ξbc
                    Ξs_kic[:,:,run,j,i] = Ξkic

                end 
            end
        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    end
end

function DiscoverSIODEModel_Ts(sys::Int, NoisePcts::Vector{Int64}, runs::Int; 
    tol=0.7, 
    c=1e-2, 
    s=1, 
    num_batches=10
    )
    @assert 1 <= sys <= 6 
    if sys == 1 # Lorenz
        # get data 
        data = load(datadir("sims", "ode_data", "lordata_si.jld"))
        data = data["dict_T"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)
        # define lib 
        function LorLib(X, ts, wind; p=10, Δ=1)
            θ = wInt_sc(X[1,:], ts, wind, p=p,Δ=Δ)
            for i=2:size(X,1)
                θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
            end
        
            for i=1:size(X,1)
                for j=i:size(X,1)
                    θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
                end
            end
            
            for i=1:size(X,1)
                for j=i:size(X,1)
                    for k=j:size(X,1)
                        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
                    end
                end
            end
            
            return θ
        end
        for T=1:length(data["ts"])
            ts = data["ts"][T]
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs 
                    println("T => $(T), NoisePct => $(NoisePct), run => $(run)")

                    # get inputs
                    qt, θ =  GetInputs(sys, data["Xtrues"][T], ts, LorLib, NoisePct)

                    Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,T] = Ξaicc
                    Ξs_slic[:,:,run,j,T] = Ξslic
                    Ξs_bic[:,:,run,j,T] = Ξbic
                    Ξs_aic[:,:,run,j,T] = Ξaic
                    Ξs_hqic[:,:,run,j,T] = Ξhqic
                    Ξs_bc[:,:,run,j,T] = Ξbc
                    Ξs_kic[:,:,run,j,T] = Ξkic

                end
            end
        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 2 # Rossler
        # get data 
        data = load(datadir("sims", "ode_data", "rossdata_si.jld"))
        data = data["dict_T"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)

         # define lib 
         function RossLib(X, ts, wind; p=10, Δ=1)
            n = size(X,1)
            θ = wInt_sc(ones(size(X[1,:])), ts, wind, p=p,Δ=Δ)
            for i=1:n
                θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
            end
        
            for i=1:n
                for j=i:n
                    θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
                end
            end
            
            for i=1:n
                for j=i:n
                    for k=j:n
                        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
                    end
                end
            end
            
            return θ
        end

        # add noise
        for T=1:length(data["ts"])
            ts = data["ts"][T]
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs
                    println("T => $(T), NoisePct => $(NoisePct), run => $(run)")

                    qt, θ = GetInputs(sys, data["Xtrues"][T], ts, RossLib, NoisePct)

                    Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,T] = Ξaicc
                    Ξs_slic[:,:,run,j,T] = Ξslic
                    Ξs_bic[:,:,run,j,T] = Ξbic
                    Ξs_aic[:,:,run,j,T] = Ξaic
                    Ξs_hqic[:,:,run,j,T] = Ξhqic
                    Ξs_bc[:,:,run,j,T] = Ξbc
                    Ξs_kic[:,:,run,j,T] = Ξkic

                end
            end
        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 3 # Lotka-Volterra
        # get data 
        data = load(datadir("sims", "ode_data", "lvdata_si.jld"))
        data = data["dict_T"]

        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)

        # define lib 
        function LVLib(X, ts, wind; p=10, Δ=1)
            n = size(X,1)
            θ = wInt_sc(X[1,:], ts, wind, p=p,Δ=Δ)
            for i=2:n
                θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
            end
        
            for i=1:n
                for j=i:n
                    θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
                end
            end
            
            for i=1:n
                for j=i:n
                    for k=j:n
                        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
                    end
                end
            end
            
            return θ
        end

        for T=1:length(data["ts"])
            ts = data["ts"][T]
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs
                    println("T => $(T), NoisePct => $(NoisePct), run => $(run)")

                    # get inputs 
                    qt, θ = GetInputs(sys, data["Xtrues"][T], ts, LVLib, NoisePct)
                    
                    # determine dynamics
                    Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,T] = Ξaicc
                    Ξs_slic[:,:,run,j,T] = Ξslic
                    Ξs_bic[:,:,run,j,T] = Ξbic
                    Ξs_aic[:,:,run,j,T] = Ξaic
                    Ξs_hqic[:,:,run,j,T] = Ξhqic
                    Ξs_bc[:,:,run,j,T] = Ξbc
                    Ξs_kic[:,:,run,j,T] = Ξkic

                end
            end
        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 4 # Brusselator
        # get data 
        data = load(datadir("sims", "ode_data", "brusdata_si.jld"))
        data = data["dict_T"]

        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)

        # define lib 
        function BrusLib(X, ts, wind; p=10, Δ=1)
            n = size(X,1)
            θ = wInt_sc(ones(size(X[1,:])), ts, wind, p=p,Δ=Δ)
            for i=1:n
                θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
            end
        
            for i=1:n
                for j=i:n
                    θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
                end
            end
            
            for i=1:n
                for j=i:n
                    for k=j:n
                        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
                    end
                end
            end
            
            return θ
        end

        for T=1:length(data["ts"])
            ts = data["ts"][T]
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs
                    println("T => $(T), NoisePct => $(NoisePct), run => $(run)")

                    # get inputs
                    qt, θ = GetInputs(sys, data["Xtrues"][T], ts, BrusLib, NoisePct)
                    
                    # discover dynamic model
                    Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,T] = Ξaicc
                    Ξs_slic[:,:,run,j,T] = Ξslic
                    Ξs_bic[:,:,run,j,T] = Ξbic
                    Ξs_aic[:,:,run,j,T] = Ξaic
                    Ξs_hqic[:,:,run,j,T] = Ξhqic
                    Ξs_bc[:,:,run,j,T] = Ξbc
                    Ξs_kic[:,:,run,j,T] = Ξkic

                end
            end
        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 5 # Van der Pol
        # get data 
        data = load(datadir("sims", "ode_data", "vdpdata_si.jld"))
        data = data["dict_T"]

        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)
        
        # define lib 
        function VdPLib(X, ts, wind; p=10, Δ=1)
            n = size(X,1)
            θ = wInt_sc(X[1,:], ts, wind, p=p,Δ=Δ)
            for i=2:n
                θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
            end
        
            for i=1:n
                for j=i:n
                    θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
                end
            end
            
            for i=1:n
                for j=i:n
                    for k=j:n
                        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
                    end
                end
            end
            
            return θ
        end

        for T=1:length(data["ts"])
            ts = data["ts"][T]
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs
                    println("T => $(T), NoisePct => $(NoisePct), run => $(run)")

                    # get inputs
                    qt, θ = GetInputs(sys, data["Xtrues"][T], ts, VdPLib, NoisePct)
                    
                    # discover model
                    Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,T] = Ξaicc
                    Ξs_slic[:,:,run,j,T] = Ξslic
                    Ξs_bic[:,:,run,j,T] = Ξbic
                    Ξs_aic[:,:,run,j,T] = Ξaic
                    Ξs_hqic[:,:,run,j,T] = Ξhqic
                    Ξs_bc[:,:,run,j,T] = Ξbc
                    Ξs_kic[:,:,run,j,T] = Ξkic

                end 
            end
        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 6 # Nonlinear Pendulum
        # get data 
        data = load(datadir("sims", "ode_data", "nlpdata_si.jld"))
        data = data["dict_T"]

        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)

        # define lib 
        function NLPLib(X, ts, wind; p=10, Δ=1)
            θ = wInt_sc(ones(length(X[:])), ts, wind, p=p, Δ=Δ)
            for i=1:5
                θ = hcat(θ, wInt_sc(sin.(i*X[:]), ts, wind, p=p, Δ=Δ))
                θ = hcat(θ, wInt_sc(cos.(i*X[:]), ts, wind, p=p, Δ=Δ))
            end
            return θ
        end

        for T=1:length(data["ts"])
            ts = data["ts"][T]
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs
                    println("T => $(T), NoisePct => $(NoisePct), run => $(run)")

                    # add noise
                    qt, θ = GetInputs(sys, data["Xtrues"][T] , ts, NLPLib, NoisePct; p=10)

                    Ξslic, _ = EnAdSR(θ, qt, "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt, "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt, "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt, "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt, "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt, "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt, "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,T] = Ξaicc
                    Ξs_slic[:,:,run,j,T] = Ξslic
                    Ξs_bic[:,:,run,j,T] = Ξbic
                    Ξs_aic[:,:,run,j,T] = Ξaic
                    Ξs_hqic[:,:,run,j,T] = Ξhqic
                    Ξs_bc[:,:,run,j,T] = Ξbc
                    Ξs_kic[:,:,run,j,T] = Ξkic

                end 
            end
        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    end
end

function DiscoverSIODEModel_dts(sys::Int, NoisePcts::Vector{Int64}, runs::Int; 
    tol=0.7, 
    c=1e-2, 
    s=1, 
    num_batches=10
    )
    @assert 1 <= sys <= 6 
    if sys == 1 # Lorenz
        # get data 
        data = load(datadir("sims", "ode_data", "lordata_si.jld"))
        data = data["dict_dt"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)
        # define lib 
        function LorLib(X, ts, wind; p=10, Δ=1)
            θ = wInt_sc(X[1,:], ts, wind, p=p,Δ=Δ)
            for i=2:size(X,1)
                θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
            end
        
            for i=1:size(X,1)
                for j=i:size(X,1)
                    θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
                end
            end
            
            for i=1:size(X,1)
                for j=i:size(X,1)
                    for k=j:size(X,1)
                        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
                    end
                end
            end
            
            return θ
        end
        for T=1:length(data["ts"])
            ts = data["ts"][T]
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs 
                    println("T => $(T), NoisePct => $(NoisePct), run => $(run)")

                    # get inputs
                    qt, θ =  GetInputs(sys, data["Xtrues"][T], ts, LorLib, NoisePct)

                    Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,T] = Ξaicc
                    Ξs_slic[:,:,run,j,T] = Ξslic
                    Ξs_bic[:,:,run,j,T] = Ξbic
                    Ξs_aic[:,:,run,j,T] = Ξaic
                    Ξs_hqic[:,:,run,j,T] = Ξhqic
                    Ξs_bc[:,:,run,j,T] = Ξbc
                    Ξs_kic[:,:,run,j,T] = Ξkic

                end
            end
        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 2 # Rossler
        # get data 
        data = load(datadir("sims", "ode_data", "rossdata_si.jld"))
        data = data["dict_dt"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)

         # define lib 
         function RossLib(X, ts, wind; p=10, Δ=1)
            n = size(X,1)
            θ = wInt_sc(ones(size(X[1,:])), ts, wind, p=p,Δ=Δ)
            for i=1:n
                θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
            end
        
            for i=1:n
                for j=i:n
                    θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
                end
            end
            
            for i=1:n
                for j=i:n
                    for k=j:n
                        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
                    end
                end
            end
            
            return θ
        end

        # add noise
        for T=1:length(data["ts"])
            ts = data["ts"][T]
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs
                    println("T => $(T), NoisePct => $(NoisePct), run => $(run)")

                    qt, θ = GetInputs(sys, data["Xtrues"][T], ts, RossLib, NoisePct)

                    Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,T] = Ξaicc
                    Ξs_slic[:,:,run,j,T] = Ξslic
                    Ξs_bic[:,:,run,j,T] = Ξbic
                    Ξs_aic[:,:,run,j,T] = Ξaic
                    Ξs_hqic[:,:,run,j,T] = Ξhqic
                    Ξs_bc[:,:,run,j,T] = Ξbc
                    Ξs_kic[:,:,run,j,T] = Ξkic

                end
            end
        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 3 # Lotka-Volterra
        # get data 
        data = load(datadir("sims", "ode_data", "lvdata_si.jld"))
        data = data["dict_dt"]

        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)

        # define lib 
        function LVLib(X, ts, wind; p=10, Δ=1)
            n = size(X,1)
            θ = wInt_sc(X[1,:], ts, wind, p=p,Δ=Δ)
            for i=2:n
                θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
            end
        
            for i=1:n
                for j=i:n
                    θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
                end
            end
            
            for i=1:n
                for j=i:n
                    for k=j:n
                        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
                    end
                end
            end
            
            return θ
        end

        for T=1:length(data["ts"])
            ts = data["ts"][T]
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs
                    println("T => $(T), NoisePct => $(NoisePct), run => $(run)")

                    # get inputs 
                    qt, θ = GetInputs(sys, data["Xtrues"][T], ts, LVLib, NoisePct)
                    
                    # determine dynamics
                    Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,T] = Ξaicc
                    Ξs_slic[:,:,run,j,T] = Ξslic
                    Ξs_bic[:,:,run,j,T] = Ξbic
                    Ξs_aic[:,:,run,j,T] = Ξaic
                    Ξs_hqic[:,:,run,j,T] = Ξhqic
                    Ξs_bc[:,:,run,j,T] = Ξbc
                    Ξs_kic[:,:,run,j,T] = Ξkic

                end
            end
        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 4 # Brusselator
        # get data 
        data = load(datadir("sims", "ode_data", "brusdata_si.jld"))
        data = data["dict_dt"]

        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)

        # define lib 
        function BrusLib(X, ts, wind; p=10, Δ=1)
            n = size(X,1)
            θ = wInt_sc(ones(size(X[1,:])), ts, wind, p=p,Δ=Δ)
            for i=1:n
                θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
            end
        
            for i=1:n
                for j=i:n
                    θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
                end
            end
            
            for i=1:n
                for j=i:n
                    for k=j:n
                        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
                    end
                end
            end
            
            return θ
        end

        for T=1:length(data["ts"])
            ts = data["ts"][T]
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs
                    println("T => $(T), NoisePct => $(NoisePct), run => $(run)")

                    # get inputs
                    qt, θ = GetInputs(sys, data["Xtrues"][T], ts, BrusLib, NoisePct)
                    
                    # discover dynamic model
                    Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,T] = Ξaicc
                    Ξs_slic[:,:,run,j,T] = Ξslic
                    Ξs_bic[:,:,run,j,T] = Ξbic
                    Ξs_aic[:,:,run,j,T] = Ξaic
                    Ξs_hqic[:,:,run,j,T] = Ξhqic
                    Ξs_bc[:,:,run,j,T] = Ξbc
                    Ξs_kic[:,:,run,j,T] = Ξkic

                end
            end
        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 5 # Van der Pol
        # get data 
        data = load(datadir("sims", "ode_data", "vdpdata_si.jld"))
        data = data["dict_dt"]

        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)
        
        # define lib 
        function VdPLib(X, ts, wind; p=10, Δ=1)
            n = size(X,1)
            θ = wInt_sc(X[1,:], ts, wind, p=p,Δ=Δ)
            for i=2:n
                θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
            end
        
            for i=1:n
                for j=i:n
                    θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
                end
            end
            
            for i=1:n
                for j=i:n
                    for k=j:n
                        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
                    end
                end
            end
            
            return θ
        end

        for T=1:length(data["ts"])
            ts = data["ts"][T]
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs
                    println("T => $(T), NoisePct => $(NoisePct), run => $(run)")

                    # get inputs
                    qt, θ = GetInputs(sys, data["Xtrues"][T], ts, VdPLib, NoisePct)
                    
                    # discover model
                    Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,T] = Ξaicc
                    Ξs_slic[:,:,run,j,T] = Ξslic
                    Ξs_bic[:,:,run,j,T] = Ξbic
                    Ξs_aic[:,:,run,j,T] = Ξaic
                    Ξs_hqic[:,:,run,j,T] = Ξhqic
                    Ξs_bc[:,:,run,j,T] = Ξbc
                    Ξs_kic[:,:,run,j,T] = Ξkic

                end 
            end
        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 6 # Nonlinear Pendulum
        # get data 
        data = load(datadir("sims", "ode_data", "nlpdata_si.jld"))
        data = data["dict_dt"]

        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)

        # define lib 
        function NLPLib(X, ts, wind; p=10, Δ=1)
            θ = wInt_sc(ones(length(X[:])), ts, wind, p=p, Δ=Δ)
            for i=1:5
                θ = hcat(θ, wInt_sc(sin.(i*X[:]), ts, wind, p=p, Δ=Δ))
                θ = hcat(θ, wInt_sc(cos.(i*X[:]), ts, wind, p=p, Δ=Δ))
            end
            return θ
        end

        for T=1:length(data["ts"])
            ts = data["ts"][T]
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs
                    println("T => $(T), NoisePct => $(NoisePct), run => $(run)")

                    # add noise
                    qt, θ = GetInputs(sys, data["Xtrues"][T] , ts, NLPLib, NoisePct; p=10)

                    Ξslic, _ = EnAdSR(θ, qt, "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt, "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt, "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt, "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt, "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt, "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt, "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,T] = Ξaicc
                    Ξs_slic[:,:,run,j,T] = Ξslic
                    Ξs_bic[:,:,run,j,T] = Ξbic
                    Ξs_aic[:,:,run,j,T] = Ξaic
                    Ξs_hqic[:,:,run,j,T] = Ξhqic
                    Ξs_bc[:,:,run,j,T] = Ξbc
                    Ξs_kic[:,:,run,j,T] = Ξkic

                end 
            end
        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    end
end

function DiscoverSIODEModel_ps(sys::Int, NoisePcts::Vector{Int64}, runs::Int; 
    tol=0.7, 
    c=1e-2, 
    s=1, 
    num_batches=10
    )
    @assert sys ∈ [1,3,4,5] 
    if sys == 1 # Lorenz
        # get data 
        data = load(datadir("sims", "ode_data", "lordata_si.jld"))
        data = data["dict_ps"]
        nps = length(data["Ξtrues"])

        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrues"][1])..., runs, length(NoisePcts), nps)
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)
        # define lib 
        function LorLib(X, ts, wind; p=10, Δ=1)
            θ = wInt_sc(X[1,:], ts, wind, p=p,Δ=Δ)
            for i=2:size(X,1)
                θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
            end
        
            for i=1:size(X,1)
                for j=i:size(X,1)
                    θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
                end
            end
            
            for i=1:size(X,1)
                for j=i:size(X,1)
                    for k=j:size(X,1)
                        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
                    end
                end
            end
            
            return θ
        end
        for T=1:nps
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs 
                    println("T => $(T), NoisePct => $(NoisePct), run => $(run)")

                    # get inputs
                    qt, θ =  GetInputs(sys, data["Xtrues"][T], data["ts"], LorLib, NoisePct)

                    Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,T] = Ξaicc
                    Ξs_slic[:,:,run,j,T] = Ξslic
                    Ξs_bic[:,:,run,j,T] = Ξbic
                    Ξs_aic[:,:,run,j,T] = Ξaic
                    Ξs_hqic[:,:,run,j,T] = Ξhqic
                    Ξs_bc[:,:,run,j,T] = Ξbc
                    Ξs_kic[:,:,run,j,T] = Ξkic

                end
            end
        end
        return data["Ξtrues"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 3 # Lotka-Volterra
        # get data 
        data = load(datadir("sims", "ode_data", "lvdata_si.jld"))
        data = data["dict_ps"]
        nps = length(data["Ξtrues"])

        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrues"][1])..., runs, length(NoisePcts), nps)
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)

        # define lib 
        function LVLib(X, ts, wind; p=10, Δ=1)
            n = size(X,1)
            θ = wInt_sc(X[1,:], ts, wind, p=p,Δ=Δ)
            for i=2:n
                θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
            end
        
            for i=1:n
                for j=i:n
                    θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
                end
            end
            
            for i=1:n
                for j=i:n
                    for k=j:n
                        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
                    end
                end
            end
            
            return θ
        end

        for T=1:nps
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs
                    println("T => $(T), NoisePct => $(NoisePct), run => $(run)")

                    # get inputs 
                    qt, θ = GetInputs(sys, data["Xtrues"][T], data["ts"], LVLib, NoisePct)
                    
                    # determine dynamics
                    Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,T] = Ξaicc
                    Ξs_slic[:,:,run,j,T] = Ξslic
                    Ξs_bic[:,:,run,j,T] = Ξbic
                    Ξs_aic[:,:,run,j,T] = Ξaic
                    Ξs_hqic[:,:,run,j,T] = Ξhqic
                    Ξs_bc[:,:,run,j,T] = Ξbc
                    Ξs_kic[:,:,run,j,T] = Ξkic

                end
            end
        end
        return data["Ξtrues"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 4 # Brusselator
        # get data 
        data = load(datadir("sims", "ode_data", "brusdata_si.jld"))
        data = data["dict_ps"]
        nps = length(data["Ξtrues"])

        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrues"][1])..., runs, length(NoisePcts), nps)
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)

        # define lib 
        function BrusLib(X, ts, wind; p=10, Δ=1)
            n = size(X,1)
            θ = wInt_sc(ones(size(X[1,:])), ts, wind, p=p,Δ=Δ)
            for i=1:n
                θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
            end
        
            for i=1:n
                for j=i:n
                    θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
                end
            end
            
            for i=1:n
                for j=i:n
                    for k=j:n
                        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
                    end
                end
            end
            
            return θ
        end

        for T=1:nps
            for j =1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs
                    println("T => $(T), NoisePct => $(NoisePct), run => $(run)")

                    # get inputs
                    qt, θ = GetInputs(sys, data["Xtrues"][T], data["ts"], BrusLib, NoisePct)
                    
                    # discover dynamic model
                    Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,T] = Ξaicc
                    Ξs_slic[:,:,run,j,T] = Ξslic
                    Ξs_bic[:,:,run,j,T] = Ξbic
                    Ξs_aic[:,:,run,j,T] = Ξaic
                    Ξs_hqic[:,:,run,j,T] = Ξhqic
                    Ξs_bc[:,:,run,j,T] = Ξbc
                    Ξs_kic[:,:,run,j,T] = Ξkic

                end
            end
        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 5 # Van der Pol
        # get data 
        data = load(datadir("sims", "ode_data", "vdpdata_si.jld"))
        data = data["dict_ps"]
        nps = length(data["Ξtrues"])

        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrues"][1])..., runs, length(NoisePcts), nps)
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)
        
        # define lib 
        function VdPLib(X, ts, wind; p=10, Δ=1)
            n = size(X,1)
            θ = wInt_sc(X[1,:], ts, wind, p=p,Δ=Δ)
            for i=2:n
                θ  = hcat(θ, wInt_sc(X[i,:], ts, wind, p=p,Δ=Δ))
            end
        
            for i=1:n
                for j=i:n
                    θ = hcat(θ, wInt_sc(X[i,:].*X[j,:], ts, wind, p=p,Δ=Δ))
                end
            end
            
            for i=1:n
                for j=i:n
                    for k=j:n
                        θ = hcat(θ, wInt_sc(X[i,:].*X[j,:].*X[k,:], ts, wind, p=p,Δ=Δ))
                    end
                end
            end
            
            return θ
        end

        for T=1:nps
            for j=1:length(NoisePcts)
                NoisePct = NoisePcts[j]
                for run=1:runs
                    println("T => $(T), NoisePct => $(NoisePct), run => $(run)")

                    # get inputs
                    qt, θ = GetInputs(sys, data["Xtrues"][T], data["ts"], VdPLib, NoisePct)
                    
                    # discover model
                    Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
                    Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
                    Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
                    Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
                    Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
                    Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
                    Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

                    Ξs_aicc[:,:,run,j,T] = Ξaicc
                    Ξs_slic[:,:,run,j,T] = Ξslic
                    Ξs_bic[:,:,run,j,T] = Ξbic
                    Ξs_aic[:,:,run,j,T] = Ξaic
                    Ξs_hqic[:,:,run,j,T] = Ξhqic
                    Ξs_bc[:,:,run,j,T] = Ξbc
                    Ξs_kic[:,:,run,j,T] = Ξkic

                end 
            end
        end
        return data["Ξtrues"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    end
end

function GetInputs(sys::Int, Xs , ts, Lib::Function, NoisePct; p=10, Δ=1)
    if sys==1 || sys==2 || sys==3 || sys==4
        Xtrue = Xs[1]
        η = NoisePct*mean(std(Xtrue, dims=2))/100  
        Xn = Xtrue + η.*randn(size(Xtrue))
        Xsm = smooth_ode(Xn)
        _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)  
        qt = dwInt_sc(Xsm, ts, wind, p=p, Δ=Δ)
        θ = Lib(Xsm, ts, wind, p=p, Δ=Δ)

        for i=2:length(Xs)
            Xtrue = Xs[i]
            η = NoisePct*mean(std(Xtrue, dims=2))/100  
            Xn = Xtrue + η.*randn(size(Xtrue))
            Xsm = smooth_ode(Xn)
            _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)  
            qt_i = dwInt_sc(Xsm, ts, wind, p=p, Δ=Δ)
            θ_i = Lib(Xsm, ts, wind, p=p, Δ=Δ)

            qt = hcat(qt, qt_i)
            θ = vcat(θ,θ_i)
        end

        return qt, θ
    elseif sys==5 # for 1-D systems where we don't know velocity (VdP, NLP)
        xtrue = Xs[1][1,:]
        η = NoisePct*std(xtrue)/100  
        xn = xtrue + η.*randn(size(xtrue))
        xsm = smooth_ode(xn)
        ts_tr, x, dx = DataWithFirstDeriv(xsm', ts, ts[2]-ts[1])
        Xsm = vcat(x, dx)
        _,_,wind = FindW(Xsm, ts_tr, ws=21:2:121, p=p)  
        qt = dwInt_sc(Xsm, ts_tr, wind, p=p, Δ=Δ)
        θ = Lib(Xsm, ts_tr, wind, p=p, Δ=Δ)

        for i=2:length(Xs)
            xtrue = Xs[i][1,:]
            η = NoisePct*std(xtrue)/100  
            xn = xtrue + η.*randn(size(xtrue))
            xsm = smooth_ode(xn)
            ts_tr, x, dx = DataWithFirstDeriv(xsm', ts, ts[2]-ts[1])
            Xsm = vcat(x, dx)
            _,_,wind = FindW(Xsm, ts_tr, ws=21:2:121, p=p)  
            qt_i = dwInt_sc(Xsm, ts_tr, wind, p=p, Δ=Δ)
            θ_i = Lib(Xsm, ts_tr, wind, p=p, Δ=Δ)

            qt = hcat(qt, qt_i)
            θ = vcat(θ,θ_i)
        end

        return qt, θ
    elseif sys==6 
        xtrue = Xs[1][1,:]
        η = NoisePct*std(xtrue)/100  
        xn = xtrue + η.*randn(size(xtrue))
        xsm = smooth_ode(xn)
        Xsm = reshape(xsm, (1, length(xsm)))
        _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)  
        qt = d2wInt_sc(Xsm, ts, wind, p=p, Δ=Δ)
        qt = reshape(qt, (length(qt),1))
        θ = Lib(Xsm, ts, wind, p=p, Δ=Δ)

        for i=2:length(Xs)
            xtrue = Xs[i][1,:]
            η = NoisePct*std(xtrue)/100  
            xn = xtrue + η.*randn(size(xtrue))
            xsm = smooth_ode(xn)
            Xsm = reshape(xsm, (1, length(xsm)))
            _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p) 
            qt_i = d2wInt_sc(Xsm, ts, wind, p=p, Δ=Δ)
            qt_i = reshape(qt_i, (length(qt_i),1))
            θ_i = Lib(Xsm, ts, wind, p=p, Δ=Δ)

            qt = vcat(qt, qt_i)
            θ = vcat(θ,θ_i)
        end

        return qt, θ
    end
end


# parameters
NoisePcts = [0, 5, 10, 20, 30, 40]
tol = 0.7
num_batches = 20 
runs = 25 


# ========== Lorenz ============
sys = 1 
c = 1e-1

println("Beginning sys #$(sys)")
# subsampling
println("Subsampling")
Ξtrue_lor, Ξaics_lor, Ξaiccs_lor, Ξslics_lor, Ξbics_lor, Ξhqics_lor, Ξbcs_lor, Ξkics_lor = DiscoverSIODEModel_ss(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_ss = Dict(
    "Ξtrue" => Ξtrue_lor, 
    "Ξaics" => Ξaics_lor,
    "Ξaiccs" => Ξaiccs_lor, 
    "Ξbics" => Ξbics_lor,
    "Ξhqics" => Ξhqics_lor,
    "Ξkics" => Ξkics_lor,
    "Ξbcs" => Ξbcs_lor,
    "Ξslics" => Ξslics_lor,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches
)


# Ts
println("Traj. Length")

Ξtrue_lor, Ξaics_lor, Ξaiccs_lor, Ξslics_lor, Ξbics_lor, Ξhqics_lor, Ξbcs_lor, Ξkics_lor = DiscoverSIODEModel_Ts(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_Ts = Dict(
    "Ξtrue" => Ξtrue_lor, 
    "Ξaics" => Ξaics_lor,
    "Ξaiccs" => Ξaiccs_lor, 
    "Ξbics" => Ξbics_lor,
    "Ξhqics" => Ξhqics_lor,
    "Ξkics" => Ξkics_lor,
    "Ξbcs" => Ξbcs_lor,
    "Ξslics" => Ξslics_lor,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches
)


# dts
println("Sampling frequency")

Ξtrue_lor, Ξaics_lor, Ξaiccs_lor, Ξslics_lor, Ξbics_lor, Ξhqics_lor, Ξbcs_lor, Ξkics_lor = DiscoverSIODEModel_dts(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_dts = Dict(
    "Ξtrue" => Ξtrue_lor, 
    "Ξaics" => Ξaics_lor,
    "Ξaiccs" => Ξaiccs_lor, 
    "Ξbics" => Ξbics_lor,
    "Ξhqics" => Ξhqics_lor,
    "Ξkics" => Ξkics_lor,
    "Ξbcs" => Ξbcs_lor,
    "Ξslics" => Ξslics_lor,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches
)
 

# ps
println("Parameters")
Ξtrue_lor, Ξaics_lor, Ξaiccs_lor, Ξslics_lor, Ξbics_lor, Ξhqics_lor, Ξbcs_lor, Ξkics_lor = DiscoverSIODEModel_ps(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_ps = Dict(
    "Ξtrues" => Ξtrue_lor, 
    "Ξaics" => Ξaics_lor,
    "Ξaiccs" => Ξaiccs_lor, 
    "Ξbics" => Ξbics_lor,
    "Ξhqics" => Ξhqics_lor,
    "Ξkics" => Ξkics_lor,
    "Ξbcs" => Ξbcs_lor,
    "Ξslics" => Ξslics_lor,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches
)


lor_results_si = Dict(
    "results_ss" => results_ss, 
    "results_Ts" => results_Ts,
    "results_dts" => results_dts,
    "results_ps" => results_ps
)

println("Saving data for sys #$(sys)")
#wsave(datadir("sims", "ode_results_si", "lor_results_si.jld2"), lor_results_si)

 
# Rossler 
sys = 2 
c=1e-1

println("Beginning sys #$(sys)")
# subsampling
println("Subsampling")
Ξtrue_ross, Ξaics_ross, Ξaiccs_ross, Ξslics_ross, Ξbics_ross, Ξhqics_ross, Ξbcs_ross, Ξkics_ross  = DiscoverSIODEModel_ss(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_ss = Dict(
    "Ξtrue" => Ξtrue_ross, 
    "Ξaics" => Ξaics_ross,
    "Ξaiccs" => Ξaiccs_ross, 
    "Ξbics" => Ξbics_ross,
    "Ξhqics" => Ξhqics_ross,
    "Ξkics" => Ξkics_ross,
    "Ξbcs" => Ξbcs_ross,
    "Ξslics" => Ξslics_ross,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# Ts
println("Traj. Length")
Ξtrue_ross, Ξaics_ross, Ξaiccs_ross, Ξslics_ross, Ξbics_ross, Ξhqics_ross, Ξbcs_ross, Ξkics_ross  = DiscoverSIODEModel_Ts(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_Ts = Dict(
    "Ξtrue" => Ξtrue_ross, 
    "Ξaics" => Ξaics_ross,
    "Ξaiccs" => Ξaiccs_ross, 
    "Ξbics" => Ξbics_ross,
    "Ξhqics" => Ξhqics_ross,
    "Ξkics" => Ξkics_ross,
    "Ξbcs" => Ξbcs_ross,
    "Ξslics" => Ξslics_ross,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# dts 
println("Sampling frequency")
Ξtrue_ross, Ξaics_ross, Ξaiccs_ross, Ξslics_ross, Ξbics_ross, Ξhqics_ross, Ξbcs_ross, Ξkics_ross  = DiscoverSIODEModel_dts(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_dts = Dict(
    "Ξtrue" => Ξtrue_ross, 
    "Ξaics" => Ξaics_ross,
    "Ξaiccs" => Ξaiccs_ross, 
    "Ξbics" => Ξbics_ross,
    "Ξhqics" => Ξhqics_ross,
    "Ξkics" => Ξkics_ross,
    "Ξbcs" => Ξbcs_ross,
    "Ξslics" => Ξslics_ross,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

ross_results_si = Dict(
    "results_ss" => results_ss, 
    "results_Ts" => results_Ts,
    "results_dts" => results_dts
)

println("Saving data for sys #$(sys)")
#wsave(datadir("sims", "ode_results_si", "ross_results_si.jld2"), ross_results_si)


# ======== L-V ============
sys = 3 
c=1e-3

println("Beginning sys #$(sys)")
# subsampling
println("Sampling")
Ξtrue_lv, Ξaics_lv, Ξaiccs_lv, Ξslics_lv, Ξbics_lv, Ξhqics_lv, Ξbcs_lv, Ξkics_lv  = DiscoverSIODEModel_ss(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_ss = Dict(
    "Ξtrue" => Ξtrue_lv, 
    "Ξaics" => Ξaics_lv,
    "Ξaiccs" => Ξaiccs_lv, 
    "Ξbics" => Ξbics_lv,
    "Ξhqics" => Ξhqics_lv,
    "Ξkics" => Ξkics_lv,
    "Ξbcs" => Ξbcs_lv,
    "Ξslics" => Ξslics_lv,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# Ts
println("Traj. Length")
Ξtrue_lv, Ξaics_lv, Ξaiccs_lv, Ξslics_lv, Ξbics_lv, Ξhqics_lv, Ξbcs_lv, Ξkics_lv  = DiscoverSIODEModel_Ts(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_Ts = Dict(
    "Ξtrue" => Ξtrue_lv, 
    "Ξaics" => Ξaics_lv,
    "Ξaiccs" => Ξaiccs_lv, 
    "Ξbics" => Ξbics_lv,
    "Ξhqics" => Ξhqics_lv,
    "Ξkics" => Ξkics_lv,
    "Ξbcs" => Ξbcs_lv,
    "Ξslics" => Ξslics_lv,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# dts 
println("Sampling frequency")
Ξtrue_lv, Ξaics_lv, Ξaiccs_lv, Ξslics_lv, Ξbics_lv, Ξhqics_lv, Ξbcs_lv, Ξkics_lv  = DiscoverSIODEModel_dts(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_dts = Dict(
    "Ξtrue" => Ξtrue_lv, 
    "Ξaics" => Ξaics_lv,
    "Ξaiccs" => Ξaiccs_lv, 
    "Ξbics" => Ξbics_lv,
    "Ξhqics" => Ξhqics_lv,
    "Ξkics" => Ξkics_lv,
    "Ξbcs" => Ξbcs_lv,
    "Ξslics" => Ξslics_lv,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# ps 
println("Parameters")
Ξtrue_lv, Ξaics_lv, Ξaiccs_lv, Ξslics_lv, Ξbics_lv, Ξhqics_lv, Ξbcs_lv, Ξkics_lv  = DiscoverSIODEModel_ps(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_ps = Dict(
    "Ξtrue" => Ξtrue_lv, 
    "Ξaics" => Ξaics_lv,
    "Ξaiccs" => Ξaiccs_lv, 
    "Ξbics" => Ξbics_lv,
    "Ξhqics" => Ξhqics_lv,
    "Ξkics" => Ξkics_lv,
    "Ξbcs" => Ξbcs_lv,
    "Ξslics" => Ξslics_lv,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

lv_results_si = Dict(
    "results_ss" => results_ss, 
    "results_Ts" => results_Ts,
    "results_dts" => results_dts,
    "results_ps" => results_ps
)

println("Saving data for sys #$(sys)")
#wsave(datadir("sims", "ode_results_si", "lv_results_si.jld2"), lv_results_si)



# ========= Brusselator ============
sys = 4 
c=1e-2

println("Beginning sys #$(sys)")
# ss
println("Sampling")
Ξtrue_brus, Ξaics_brus, Ξaiccs_brus, Ξslics_brus, Ξbics_brus, Ξhqics_brus, Ξbcs_brus, Ξkics_brus  = DiscoverSIODEModel_ss(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_ss = Dict(
    "Ξtrue" => Ξtrue_brus, 
    "Ξaics" => Ξaics_brus,
    "Ξaiccs" => Ξaiccs_brus, 
    "Ξbics" => Ξbics_brus,
    "Ξhqics" => Ξhqics_brus,
    "Ξkics" => Ξkics_brus,
    "Ξbcs" => Ξbcs_brus,
    "Ξslics" => Ξslics_brus,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# Ts
println("Traj. Length")
Ξtrue_brus, Ξaics_brus, Ξaiccs_brus, Ξslics_brus, Ξbics_brus, Ξhqics_brus, Ξbcs_brus, Ξkics_brus  = DiscoverSIODEModel_Ts(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_Ts = Dict(
    "Ξtrue" => Ξtrue_brus, 
    "Ξaics" => Ξaics_brus,
    "Ξaiccs" => Ξaiccs_brus, 
    "Ξbics" => Ξbics_brus,
    "Ξhqics" => Ξhqics_brus,
    "Ξkics" => Ξkics_brus,
    "Ξbcs" => Ξbcs_brus,
    "Ξslics" => Ξslics_brus,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# dts
println("Sampling frequency")
Ξtrue_brus, Ξaics_brus, Ξaiccs_brus, Ξslics_brus, Ξbics_brus, Ξhqics_brus, Ξbcs_brus, Ξkics_brus  = DiscoverSIODEModel_dts(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_dts = Dict(
    "Ξtrue" => Ξtrue_brus, 
    "Ξaics" => Ξaics_brus,
    "Ξaiccs" => Ξaiccs_brus, 
    "Ξbics" => Ξbics_brus,
    "Ξhqics" => Ξhqics_brus,
    "Ξkics" => Ξkics_brus,
    "Ξbcs" => Ξbcs_brus,
    "Ξslics" => Ξslics_brus,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)


# ps
println("Parameters")
Ξtrue_brus, Ξaics_brus, Ξaiccs_brus, Ξslics_brus, Ξbics_brus, Ξhqics_brus, Ξbcs_brus, Ξkics_brus  = DiscoverSIODEModel_ps(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_ps = Dict(
    "Ξtrue" => Ξtrue_brus, 
    "Ξaics" => Ξaics_brus,
    "Ξaiccs" => Ξaiccs_brus, 
    "Ξbics" => Ξbics_brus,
    "Ξhqics" => Ξhqics_brus,
    "Ξkics" => Ξkics_brus,
    "Ξbcs" => Ξbcs_brus,
    "Ξslics" => Ξslics_brus,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

brus_results_si = Dict(
    "results_ss" => results_ss, 
    "results_Ts" => results_Ts,
    "results_dts" => results_dts,
    "results_ps" => results_ps
)

println("Saving data for sys #$(sys)")
#wsave(datadir("sims", "ode_results_si", "brus_results_si.jld2"), brus_results_si)
 


# ======== VdP =============
sys = 5 
c=1e-0

println("Beginning sys #$(sys)")
# ss 
println("Sampling")
Ξtrue_vdp, Ξaics_vdp, Ξaiccs_vdp, Ξslics_vdp, Ξbics_vdp, Ξhqics_vdp, Ξbcs_vdp, Ξkics_vdp  = DiscoverSIODEModel_ss(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_ss = Dict(
    "Ξtrue" => Ξtrue_vdp, 
    "Ξaics" => Ξaics_vdp,
    "Ξaiccs" => Ξaiccs_vdp, 
    "Ξbics" => Ξbics_vdp,
    "Ξhqics" => Ξhqics_vdp,
    "Ξkics" => Ξkics_vdp,
    "Ξbcs" => Ξbcs_vdp,
    "Ξslics" => Ξslics_vdp,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# Ts 
println("Traj. Length")
Ξtrue_vdp, Ξaics_vdp, Ξaiccs_vdp, Ξslics_vdp, Ξbics_vdp, Ξhqics_vdp, Ξbcs_vdp, Ξkics_vdp  = DiscoverSIODEModel_Ts(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_Ts = Dict(
    "Ξtrue" => Ξtrue_vdp, 
    "Ξaics" => Ξaics_vdp,
    "Ξaiccs" => Ξaiccs_vdp, 
    "Ξbics" => Ξbics_vdp,
    "Ξhqics" => Ξhqics_vdp,
    "Ξkics" => Ξkics_vdp,
    "Ξbcs" => Ξbcs_vdp,
    "Ξslics" => Ξslics_vdp,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# dts 
println("Sampling frequency")
Ξtrue_vdp, Ξaics_vdp, Ξaiccs_vdp, Ξslics_vdp, Ξbics_vdp, Ξhqics_vdp, Ξbcs_vdp, Ξkics_vdp  = DiscoverSIODEModel_dts(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_dts = Dict(
    "Ξtrue" => Ξtrue_vdp, 
    "Ξaics" => Ξaics_vdp,
    "Ξaiccs" => Ξaiccs_vdp, 
    "Ξbics" => Ξbics_vdp,
    "Ξhqics" => Ξhqics_vdp,
    "Ξkics" => Ξkics_vdp,
    "Ξbcs" => Ξbcs_vdp,
    "Ξslics" => Ξslics_vdp,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# ps 
println("Parameters")
Ξtrue_vdp, Ξaics_vdp, Ξaiccs_vdp, Ξslics_vdp, Ξbics_vdp, Ξhqics_vdp, Ξbcs_vdp, Ξkics_vdp  = DiscoverSIODEModel_ps(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_ps = Dict(
    "Ξtrue" => Ξtrue_vdp, 
    "Ξaics" => Ξaics_vdp,
    "Ξaiccs" => Ξaiccs_vdp, 
    "Ξbics" => Ξbics_vdp,
    "Ξhqics" => Ξhqics_vdp,
    "Ξkics" => Ξkics_vdp,
    "Ξbcs" => Ξbcs_vdp,
    "Ξslics" => Ξslics_vdp,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

vdp_results_si = Dict(
    "results_ss" => results_ss, 
    "results_Ts" => results_Ts,
    "results_dts" => results_dts,
    "results_ps" => results_ps
)

println("Saving data for sys #$(sys)")
#wsave(datadir("sims", "ode_results_si", "vdp_results_si.jld2"), vdp_results_si)

 

# =========== NLP ===========
sys = 6 
c=1e-2

println("Beginning sys #$(sys)")
## ss
println("Sampling")
Ξtrue_nlp, Ξaics_nlp, Ξaiccs_nlp, Ξslics_nlp, Ξbics_nlp, Ξhqics_nlp, Ξbcs_nlp, Ξkics_nlp  = DiscoverSIODEModel_ss(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_ss = Dict(
    "Ξtrue" => Ξtrue_nlp, 
    "Ξaics" => Ξaics_nlp,
    "Ξaiccs" => Ξaiccs_nlp, 
    "Ξbics" => Ξbics_nlp,
    "Ξhqics" => Ξhqics_nlp,
    "Ξkics" => Ξkics_nlp,
    "Ξbcs" => Ξbcs_nlp,
    "Ξslics" => Ξslics_nlp,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)


# Ts
println("Traj. Length")
Ξtrue_nlp, Ξaics_nlp, Ξaiccs_nlp, Ξslics_nlp, Ξbics_nlp, Ξhqics_nlp, Ξbcs_nlp, Ξkics_nlp  = DiscoverSIODEModel_Ts(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_Ts = Dict(
    "Ξtrue" => Ξtrue_nlp, 
    "Ξaics" => Ξaics_nlp,
    "Ξaiccs" => Ξaiccs_nlp, 
    "Ξbics" => Ξbics_nlp,
    "Ξhqics" => Ξhqics_nlp,
    "Ξkics" => Ξkics_nlp,
    "Ξbcs" => Ξbcs_nlp,
    "Ξslics" => Ξslics_nlp,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# dts
println("Sampling frequency")
Ξtrue_nlp, Ξaics_nlp, Ξaiccs_nlp, Ξslics_nlp, Ξbics_nlp, Ξhqics_nlp, Ξbcs_nlp, Ξkics_nlp  = DiscoverSIODEModel_dts(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_dts = Dict(
    "Ξtrue" => Ξtrue_nlp, 
    "Ξaics" => Ξaics_nlp,
    "Ξaiccs" => Ξaiccs_nlp, 
    "Ξbics" => Ξbics_nlp,
    "Ξhqics" => Ξhqics_nlp,
    "Ξkics" => Ξkics_nlp,
    "Ξbcs" => Ξbcs_nlp,
    "Ξslics" => Ξslics_nlp,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)


nlp_results_si = Dict(
    "results_ss" => results_ss, 
    "results_Ts" => results_Ts,
    "results_dts" => results_dts
)

println("Saving data for sys #$(sys)")
#wsave(datadir("sims", "ode_results_si", "nlp_results_si.jld2"), nlp_results_si)
