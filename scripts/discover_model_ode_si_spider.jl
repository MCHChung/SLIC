using DrWatson
@quickactivate "SLIC"
using JLD

# functions to extract models and visualize results
include(srcdir("spider.jl"))
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
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(ss))
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1,size(data["Ξtrue"],2), runs, length(NoisePcts), length(ss))

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

                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)

                    Ξs_sp[:,:,run,j,i] = Ξsp
                    Ξs_sp_sc[:,:,run,j,i] = Ξsp_sc
                    
                end
            end
        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 2 # Rossler
        # get data 
        data = load(datadir("sims", "ode_data", "rossdata_si.jld"))
        data = data["dict_main"]
        # arrays to hold model outputs
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(ss))
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1,size(data["Ξtrue"],2), runs, length(NoisePcts), length(ss))
        

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

                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)

                    Ξs_sp[:,:,run,j,i] = Ξsp
                    Ξs_sp_sc[:,:,run,j,i] = Ξsp_sc

                end
            end
        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 3 # Lotka-Volterra
        # get data 
        data = load(datadir("sims", "ode_data", "lvdata_si.jld"))
        data = data["dict_main"]

        # arrays to hold model outputs
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(ss))
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1,size(data["Ξtrue"],2), runs, length(NoisePcts), length(ss))

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
                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)

                    Ξs_sp[:,:,run,j,i] = Ξsp
                    Ξs_sp_sc[:,:,run,j,i] = Ξsp_sc

                end
            end
        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 4 # Brusselator
        # get data 
        data = load(datadir("sims", "ode_data", "brusdata_si.jld"))
        data = data["dict_main"]

        # arrays to hold model outputs
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(ss))
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1,size(data["Ξtrue"],2), runs, length(NoisePcts), length(ss))

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
                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)

                    Ξs_sp[:,:,run,j,i] = Ξsp
                    Ξs_sp_sc[:,:,run,j,i] = Ξsp_sc

                end
            end
        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 5 # Van der Pol
        # get data 
        data = load(datadir("sims", "ode_data", "vdpdata_si.jld"))
        data = data["dict_main"]

        # arrays to hold model outputs
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(ss))
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1,size(data["Ξtrue"],2), runs, length(NoisePcts), length(ss))
        
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
                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)

                    Ξs_sp[:,:,run,j,i] = Ξsp
                    Ξs_sp_sc[:,:,run,j,i] = Ξsp_sc

                end 
            end
        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 6 # Nonlinear Pendulum
        # get data 
        data = load(datadir("sims", "ode_data", "nlpdata_si.jld"))
        data = data["dict_main"]

        # arrays to hold model outputs
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(ss))
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1,size(data["Ξtrue"],2), runs, length(NoisePcts), length(ss))

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

                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt')

                    Ξs_sp[:,:,run,j,i] = Ξsp
                    Ξs_sp_sc[:,:,run,j,i] = Ξsp_sc

                end 
            end
        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
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
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1,size(data["Ξtrue"],2), runs, length(NoisePcts), length(data["ts"]))

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

                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)

                    Ξs_sp[:,:,run,j, T] = Ξsp
                    Ξs_sp_sc[:,:,run,j,T] = Ξsp_sc


                end
            end
        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 2 # Rossler
        # get data 
        data = load(datadir("sims", "ode_data", "rossdata_si.jld"))
        data = data["dict_T"]
        # arrays to hold model outputs
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1,size(data["Ξtrue"],2), runs, length(NoisePcts), length(data["ts"]))

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

                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)

                    Ξs_sp[:,:,run,j,T] = Ξsp
                    Ξs_sp_sc[:,:,run,j,T] = Ξsp_sc

                end
            end
        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 3 # Lotka-Volterra
        # get data 
        data = load(datadir("sims", "ode_data", "lvdata_si.jld"))
        data = data["dict_T"]

        # arrays to hold model outputs
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1,size(data["Ξtrue"],2), runs, length(NoisePcts), length(data["ts"]))

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
                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)

                    Ξs_sp[:,:,run,j,T] = Ξsp
                    Ξs_sp_sc[:,:,run,j,T] = Ξsp_sc

                end
            end
        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 4 # Brusselator
        # get data 
        data = load(datadir("sims", "ode_data", "brusdata_si.jld"))
        data = data["dict_T"]

        # arrays to hold model outputs
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1,size(data["Ξtrue"],2), runs, length(NoisePcts), length(data["ts"]))

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
                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)

                    Ξs_sp[:,:,run,j,T] = Ξsp
                    Ξs_sp_sc[:,:,run,j,T] = Ξsp_sc

                end
            end
        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 5 # Van der Pol
        # get data 
        data = load(datadir("sims", "ode_data", "vdpdata_si.jld"))
        data = data["dict_T"]

        # arrays to hold model outputs
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1,size(data["Ξtrue"],2), runs, length(NoisePcts), length(data["ts"]))
        
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
                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)

                    Ξs_sp[:,:,run,j,T] = Ξsp
                    Ξs_sp_sc[:,:,run,j,T] = Ξsp_sc

                end 
            end
        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 6 # Nonlinear Pendulum
        # get data 
        data = load(datadir("sims", "ode_data", "nlpdata_si.jld"))
        data = data["dict_T"]

        # arrays to hold model outputs
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1,size(data["Ξtrue"],2), runs, length(NoisePcts), length(data["ts"]))

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

                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt')

                    Ξs_sp[:,:,run,j,T] = Ξsp
                    Ξs_sp_sc[:,:,run,j,T] = Ξsp_sc

                end 
            end
        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
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
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1,size(data["Ξtrue"],2), runs, length(NoisePcts), length(data["ts"]))

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

                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)

                    Ξs_sp[:,:,run,j,T] = Ξsp
                    Ξs_sp_sc[:,:,run,j,T] = Ξsp_sc

                end
            end
        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 2 # Rossler
        # get data 
        data = load(datadir("sims", "ode_data", "rossdata_si.jld"))
        data = data["dict_dt"]
        # arrays to hold model outputs
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1,size(data["Ξtrue"],2), runs, length(NoisePcts), length(data["ts"]))

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

                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)

                    Ξs_sp[:,:,run,j,T] = Ξsp
                    Ξs_sp_sc[:,:,run,j,T] = Ξsp_sc

                end
            end
        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 3 # Lotka-Volterra
        # get data 
        data = load(datadir("sims", "ode_data", "lvdata_si.jld"))
        data = data["dict_dt"]

        # arrays to hold model outputs
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1,size(data["Ξtrue"],2), runs, length(NoisePcts), length(data["ts"]))

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
                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)

                    Ξs_sp[:,:,run,j,T] = Ξsp
                    Ξs_sp_sc[:,:,run,j,T] = Ξsp_sc

                end
            end
        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 4 # Brusselator
        # get data 
        data = load(datadir("sims", "ode_data", "brusdata_si.jld"))
        data = data["dict_dt"]

        # arrays to hold model outputs
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1,size(data["Ξtrue"],2), runs, length(NoisePcts), length(data["ts"]))

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
                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)

                    Ξs_sp[:,:,run,j,T] = Ξsp
                    Ξs_sp_sc[:,:,run,j,T] = Ξsp_sc

                end
            end
        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 5 # Van der Pol
        # get data 
        data = load(datadir("sims", "ode_data", "vdpdata_si.jld"))
        data = data["dict_dt"]

        # arrays to hold model outputs
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1,size(data["Ξtrue"],2), runs, length(NoisePcts), length(data["ts"]))
        
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
                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)

                    Ξs_sp[:,:,run,j,T] = Ξsp
                    Ξs_sp_sc[:,:,run,j,T] = Ξsp_sc

                end 
            end
        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 6 # Nonlinear Pendulum
        # get data 
        data = load(datadir("sims", "ode_data", "nlpdata_si.jld"))
        data = data["dict_dt"]

        # arrays to hold model outputs
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs, length(NoisePcts), length(data["ts"]))
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1,size(data["Ξtrue"],2), runs, length(NoisePcts), length(data["ts"]))

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

                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt')

                    Ξs_sp[:,:,run,j,T] = Ξsp
                    Ξs_sp_sc[:,:,run,j,T] = Ξsp_sc

                end 
            end
        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
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
        Ξs_sp_sc = zeros(size(data["Ξtrues"][1])..., runs, length(NoisePcts), nps)
        Ξs_sp = zeros(size(data["Ξtrues"][1],1)+1,size(data["Ξtrues"][1],2), runs, length(NoisePcts), nps)
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

                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)

                    Ξs_sp[:,:,run,j,T] = Ξsp
                    Ξs_sp_sc[:,:,run,j,T] = Ξsp_sc

                end
            end
        end
        return data["Ξtrues"], Ξs_sp, Ξs_sp_sc
    elseif sys == 3 # Lotka-Volterra
        # get data 
        data = load(datadir("sims", "ode_data", "lvdata_si.jld"))
        data = data["dict_ps"]
        nps = length(data["Ξtrues"])

        # arrays to hold model outputs
        Ξs_sp_sc = zeros(size(data["Ξtrues"][1])..., runs, length(NoisePcts), nps)
        Ξs_sp = zeros(size(data["Ξtrues"][1],1)+1,size(data["Ξtrues"][1],2), runs, length(NoisePcts), nps)

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
                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)

                    Ξs_sp[:,:,run,j,T] = Ξsp
                    Ξs_sp_sc[:,:,run,j,T] = Ξsp_sc

                end
            end
        end
        return data["Ξtrues"], Ξs_sp, Ξs_sp_sc
    elseif sys == 4 # Brusselator
        # get data 
        data = load(datadir("sims", "ode_data", "brusdata_si.jld"))
        data = data["dict_ps"]
        nps = length(data["Ξtrues"])

        # arrays to hold model outputs
        Ξs_sp_sc = zeros(size(data["Ξtrues"][1])..., runs, length(NoisePcts), nps)
        Ξs_sp = zeros(size(data["Ξtrues"][1],1)+1,size(data["Ξtrues"][1],2), runs, length(NoisePcts), nps)

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
                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)

                    Ξs_sp[:,:,run,j,T] = Ξsp
                    Ξs_sp_sc[:,:,run,j,T] = Ξsp_sc

                end
            end
        end
        return data["Ξtrues"], Ξs_sp, Ξs_sp_sc
    elseif sys == 5 # Van der Pol
        # get data 
        data = load(datadir("sims", "ode_data", "vdpdata_si.jld"))
        data = data["dict_ps"]
        nps = length(data["Ξtrues"])

        # arrays to hold model outputs
        Ξs_sp_sc = zeros(size(data["Ξtrues"][1])..., runs, length(NoisePcts), nps)
        Ξs_sp = zeros(size(data["Ξtrues"][1],1)+1, size(data["Ξtrues"][1],2), runs, length(NoisePcts), nps)
        
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
                    Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)

                    Ξs_sp[:,:,run,j,T] = Ξsp
                    Ξs_sp_sc[:,:,run,j,T] = Ξsp_sc

                end 
            end
        end
        return data["Ξtrues"], Ξs_sp, Ξs_sp_sc
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
Ξtrue_lor, Ξsp_lor, Ξsp_sc_lor = DiscoverSIODEModel_ss(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_ss = Dict(
    "Ξtrue" => Ξtrue_lor, 
    "Ξsp" => Ξsp_lor,
    "Ξsp_sc" => Ξsp_sc_lor, 
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches
)


# Ts
println("Traj. Length")

Ξtrue_lor, Ξsp_lor, Ξsp_sc_lor = DiscoverSIODEModel_Ts(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_Ts = Dict(
    "Ξtrue" => Ξtrue_lor, 
    "Ξsp" => Ξsp_lor,
    "Ξsp_sc" => Ξsp_sc_lor,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches
)


# dts
println("Sampling frequency")

Ξtrue_lor, Ξsp_lor, Ξsp_sc_lor = DiscoverSIODEModel_dts(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_dts = Dict(
    "Ξtrue" => Ξtrue_lor, 
    "Ξsp" => Ξsp_lor,
    "Ξsp_sc" => Ξsp_sc_lor,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches
)
 

# ps
println("Parameters")
Ξtrue_lor, Ξsp_lor, Ξsp_sc_lor = DiscoverSIODEModel_ps(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_ps = Dict(
    "Ξtrues" => Ξtrue_lor, 
    "Ξsp" => Ξsp_lor,
    "Ξsp_sc" => Ξsp_sc_lor,
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
wsave(datadir("sims", "ode_results_si", "lor_results_spider_si.jld2"), lor_results_si)

 
# Rossler 
sys = 2 
c=1e-1

println("Beginning sys #$(sys)")
# subsampling
println("Subsampling")
Ξtrue_ross, Ξsp_ross, Ξsp_sc_ross  = DiscoverSIODEModel_ss(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_ss = Dict(
    "Ξtrue" => Ξtrue_ross, 
    "Ξsp" => Ξsp_ross,
    "Ξsp_sc" => Ξsp_sc_ross,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# Ts
println("Traj. Length")
Ξtrue_ross, Ξsp_ross, Ξsp_sc_ross  = DiscoverSIODEModel_Ts(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_Ts = Dict(
    "Ξtrue" => Ξtrue_ross, 
    "Ξsp" => Ξsp_ross,
    "Ξsp_sc" => Ξsp_sc_ross,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# dts 
println("Sampling frequency")
Ξtrue_ross, Ξsp_ross, Ξsp_sc_ross  = DiscoverSIODEModel_dts(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_dts = Dict(
    "Ξtrue" => Ξtrue_ross, 
    "Ξsp" => Ξsp_ross,
    "Ξsp_sc" => Ξsp_sc_ross,
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
wsave(datadir("sims", "ode_results_si", "ross_results_spider_si.jld2"), ross_results_si)


# ======== L-V ============
sys = 3 
c=1e-3

println("Beginning sys #$(sys)")
# subsampling
println("Sampling")
Ξtrue_lv, Ξsp_lv, Ξsp_sc_lv  = DiscoverSIODEModel_ss(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_ss = Dict(
    "Ξtrue" => Ξtrue_lv, 
    "Ξsp" => Ξsp_lv,
    "Ξsp_sc" => Ξsp_sc_lv,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# Ts
println("Traj. Length")
Ξtrue_lv, Ξsp_lv, Ξsp_sc_lv  = DiscoverSIODEModel_Ts(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_Ts = Dict(
    "Ξtrue" => Ξtrue_lv, 
    "Ξsp" => Ξsp_lv,
    "Ξsp_sc" => Ξsp_sc_lv,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# dts 
println("Sampling frequency")
Ξtrue_lv, Ξsp_lv, Ξsp_sc_lv  = DiscoverSIODEModel_dts(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_dts = Dict(
    "Ξtrue" => Ξtrue_lv, 
    "Ξsp" => Ξsp_lv,
    "Ξsp_sc" => Ξsp_sc_lv,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# ps 
println("Parameters")
Ξtrue_lv, Ξsp_lv, Ξsp_sc_lv  = DiscoverSIODEModel_ps(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_ps = Dict(
    "Ξtrue" => Ξtrue_lv, 
    "Ξsp" => Ξsp_lv,
    "Ξsp_sc" => Ξsp_sc_lv,
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
wsave(datadir("sims", "ode_results_si", "lv_results_spider_si.jld2"), lv_results_si)



# ========= Brusselator ============
sys = 4 
c=1e-2

println("Beginning sys #$(sys)")
# ss
println("Sampling")
Ξtrue_brus, Ξsp_brus, Ξsp_sc_brus   = DiscoverSIODEModel_ss(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_ss = Dict(
    "Ξtrue" => Ξtrue_brus, 
    "Ξsp" => Ξsp_brus,
    "Ξsp_sc" => Ξsp_sc_brus,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# Ts
println("Traj. Length")
Ξtrue_brus, Ξsp_brus, Ξsp_sc_brus  = DiscoverSIODEModel_Ts(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_Ts = Dict(
    "Ξtrue" => Ξtrue_brus, 
    "Ξsp" => Ξsp_brus,
    "Ξsp_sc" => Ξsp_sc_brus,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# dts
println("Sampling frequency")
Ξtrue_brus, Ξsp_brus, Ξsp_sc_brus  = DiscoverSIODEModel_dts(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_dts = Dict(
    "Ξtrue" => Ξtrue_brus, 
    "Ξsp" => Ξsp_brus,
    "Ξsp_sc" => Ξsp_sc_brus,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)


# ps
println("Parameters")
Ξtrue_brus, Ξsp_brus, Ξsp_sc_brus  = DiscoverSIODEModel_ps(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_ps = Dict(
    "Ξtrue" => Ξtrue_brus, 
    "Ξsp" => Ξsp_brus,
    "Ξsp_sc" => Ξsp_sc_brus,
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
wsave(datadir("sims", "ode_results_si", "brus_results_spider_si.jld2"), brus_results_si)
 


# ======== VdP =============
sys = 5 
c=1e-0

println("Beginning sys #$(sys)")
# ss 
println("Sampling")
Ξtrue_vdp, Ξsp_vdp, Ξsp_sc_vdp = DiscoverSIODEModel_ss(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_ss = Dict(
    "Ξtrue" => Ξtrue_vdp, 
    "Ξsp" => Ξsp_vdp,
    "Ξsp_sc" => Ξsp_sc_vdp,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# Ts 
println("Traj. Length")
Ξtrue_vdp, Ξsp_vdp, Ξsp_sc_vdp  = DiscoverSIODEModel_Ts(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_Ts = Dict(
    "Ξtrue" => Ξtrue_vdp, 
    "Ξsp" => Ξsp_vdp,
    "Ξsp_sc" => Ξsp_sc_vdp,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# dts 
println("Sampling frequency")
Ξtrue_vdp, Ξsp_vdp, Ξsp_sc_vdp  = DiscoverSIODEModel_dts(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_dts = Dict(
    "Ξtrue" => Ξtrue_vdp, 
    "Ξsp" => Ξsp_vdp,
    "Ξsp_sc" => Ξsp_sc_vdp,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# ps 
println("Parameters")
Ξtrue_vdp, Ξsp_vdp, Ξsp_sc_vdp  = DiscoverSIODEModel_ps(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)


results_ps = Dict(
    "Ξtrue" => Ξtrue_vdp, 
    "Ξsp" => Ξsp_vdp,
    "Ξsp_sc" => Ξsp_sc_vdp,
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
wsave(datadir("sims", "ode_results_si", "vdp_results_spider_si.jld2"), vdp_results_si)

 

# =========== NLP ===========
sys = 6 
c=1e-2

println("Beginning sys #$(sys)")
## ss
println("Sampling")
Ξtrue_nlp, Ξsp_nlp, Ξsp_sc_nlp   = DiscoverSIODEModel_ss(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_ss = Dict(
    "Ξtrue" => Ξtrue_nlp, 
    "Ξsp" => Ξsp_nlp,
    "Ξsp_sc" => Ξsp_sc_nlp,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)


# Ts
println("Traj. Length")
Ξtrue_nlp, Ξsp_nlp, Ξsp_sc_nlp  = DiscoverSIODEModel_Ts(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_Ts = Dict(
    "Ξtrue" => Ξtrue_nlp, 
    "Ξsp" => Ξsp_nlp,
    "Ξsp_sc" => Ξsp_sc_nlp,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# dts
println("Sampling frequency")
Ξtrue_nlp, Ξsp_nlp, Ξsp_sc_nlp  = DiscoverSIODEModel_dts(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

results_dts = Dict(
    "Ξtrue" => Ξtrue_nlp, 
    "Ξsp" => Ξsp_nlp,
    "Ξsp_sc" => Ξsp_sc_nlp,
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
wsave(datadir("sims", "ode_results_si", "nlp_results_spider_si.jld2"), nlp_results_si)

