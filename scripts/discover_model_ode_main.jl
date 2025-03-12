using DrWatson
@quickactivate "SLIC"
using JLD

# functions to extract models and visualize results
include(srcdir("sparse_regress.jl"))
include(srcdir("derivative.jl"))
include(srcdir("galerkin_proj.jl"))
include(srcdir("smooth.jl"))
include(srcdir("vis_results.jl"))

# all the ode data 
files = readdir(datadir("sims"))

# INPUTS: 
# sys = system to perform model discovery on 
# NoisePct = % noise defined as std(noise)/100*std(data)
# runs = #  times we randomly instantiate the noise at fixed noise %
# tol = prob thresholding 
# c = helps avoids log(MSE) -> infinity as MSE -> 0, (see sparse_regress.jl)
# s = subsampling index (size of jump in sliding window), s=1 indicates no subsampling 

# OUTPUTS: 
# Ξtrue = true model coefficent 
# Ξs = ic-determined models  

function DiscoverODEModel(sys::Int, NoisePct::Int64, runs::Int; 
    tol=0.7, 
    c=1e-2, 
    s=1, 
    num_batches=10
    )
    @assert 1 <= sys <= 6 && NoisePct >=0
    if sys == 1 # Lorenz
        # get data 
        data = load(datadir("sims", "ode_data", "lordata.jld"))
        
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
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

        for run=1:runs
            
            # get inputs
            qt, θ =  GetInputs(sys, data["Xtrues"], data["ts"], LorLib, NoisePct)

            Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
            Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
            Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
            Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
            Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
            Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
            Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

            Ξs_aicc[:,:,run] = Ξaicc
            Ξs_slic[:,:,run] = Ξslic
            Ξs_bic[:,:,run] = Ξbic
            Ξs_aic[:,:,run] = Ξaic
            Ξs_hqic[:,:,run] = Ξhqic
            Ξs_bc[:,:,run] = Ξbc
            Ξs_kic[:,:,run] = Ξkic

        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 2 # Rossler
        # get data 
        data = load(datadir("sims", "ode_data", "rossdata.jld"))
        
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
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
        for run=1:runs
            qt, θ = GetInputs(sys, data["Xtrues"], data["ts"], RossLib, NoisePct)

            Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
            Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
            Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
            Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
            Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
            Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
            Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

            Ξs_aicc[:,:,run] = Ξaicc
            Ξs_slic[:,:,run] = Ξslic
            Ξs_bic[:,:,run] = Ξbic
            Ξs_aic[:,:,run] = Ξaic
            Ξs_hqic[:,:,run] = Ξhqic
            Ξs_bc[:,:,run] = Ξbc
            Ξs_kic[:,:,run] = Ξkic

        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 3 # Lotka-Volterra
        # get data 
        data = load(datadir("sims", "ode_data", "lvdata.jld"))
        
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
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

        for run=1:runs
            # get inputs 
            qt, θ = GetInputs(sys, data["Xtrues"], data["ts"], LVLib, NoisePct)
            
            # determine dynamics
            Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
            Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
            Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
            Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
            Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
            Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
            Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

            Ξs_aicc[:,:,run] = Ξaicc
            Ξs_slic[:,:,run] = Ξslic
            Ξs_bic[:,:,run] = Ξbic
            Ξs_aic[:,:,run] = Ξaic
            Ξs_hqic[:,:,run] = Ξhqic
            Ξs_bc[:,:,run] = Ξbc
            Ξs_kic[:,:,run] = Ξkic

        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 4 # Brusselator
        # get data 
        data = load(datadir("sims", "ode_data", "brusdata.jld"))
        
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
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

        for run=1:runs
            # get inputs
            qt, θ = GetInputs(sys, data["Xtrues"], data["ts"], BrusLib, NoisePct)
            
            # discover dynamic model
            Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
            Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
            Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
            Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
            Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
            Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
            Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

            Ξs_aicc[:,:,run] = Ξaicc
            Ξs_slic[:,:,run] = Ξslic
            Ξs_bic[:,:,run] = Ξbic
            Ξs_aic[:,:,run] = Ξaic
            Ξs_hqic[:,:,run] = Ξhqic
            Ξs_bc[:,:,run] = Ξbc
            Ξs_kic[:,:,run] = Ξkic

        end
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 5 # Van der Pol
        # get data 
        data = load(datadir("sims", "ode_data", "vdpdata.jld"))
        
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
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

        for run=1:runs
            # get inputs
            qt, θ = GetInputs(sys, data["Xtrues"], data["ts"], VdPLib, NoisePct)
            
            # discover model
            Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
            Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)
            Ξaic, _ = EnAdSR(θ, qt', "aic", tol=tol, c=c, num_batches=num_batches)
            Ξbic, _ = EnAdSR(θ, qt', "bic", tol=tol, c=c, num_batches=num_batches)
            Ξhqic, _ = EnAdSR(θ, qt', "hqic", tol=tol, c=c, num_batches=num_batches)
            Ξbc, _ = EnAdSR(θ, qt', "bc", tol=tol, c=c, num_batches=num_batches)
            Ξkic, _ = EnAdSR(θ, qt', "kic", tol=tol, c=c, num_batches=num_batches)

            Ξs_aicc[:,:,run] = Ξaicc
            Ξs_slic[:,:,run] = Ξslic
            Ξs_bic[:,:,run] = Ξbic
            Ξs_aic[:,:,run] = Ξaic
            Ξs_hqic[:,:,run] = Ξhqic
            Ξs_bc[:,:,run] = Ξbc
            Ξs_kic[:,:,run] = Ξkic

        end 
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    elseif sys == 6 # Nonlinear Pendulum
        # get data 
        data = load(datadir("sims", "ode_data", "nlpdata.jld"))
        
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
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

        for run=1:runs
            # add noise
            qt, θ = GetInputs(sys, data["Xtrues"] , data["ts"], NLPLib, NoisePct; p=10)

            Ξslic, _ = EnAdSR(θ, qt, "slic", tol=tol, c=c, num_batches=num_batches)
            Ξaicc, _ = EnAdSR(θ, qt, "aicc", tol=tol, c=c, num_batches=num_batches)
            Ξaic, _ = EnAdSR(θ, qt, "aic", tol=tol, c=c, num_batches=num_batches)
            Ξbic, _ = EnAdSR(θ, qt, "bic", tol=tol, c=c, num_batches=num_batches)
            Ξhqic, _ = EnAdSR(θ, qt, "hqic", tol=tol, c=c, num_batches=num_batches)
            Ξbc, _ = EnAdSR(θ, qt, "bc", tol=tol, c=c, num_batches=num_batches)
            Ξkic, _ = EnAdSR(θ, qt, "kic", tol=tol, c=c, num_batches=num_batches)

            Ξs_aicc[:,:,run] = Ξaicc
            Ξs_slic[:,:,run] = Ξslic
            Ξs_bic[:,:,run] = Ξbic
            Ξs_aic[:,:,run] = Ξaic
            Ξs_hqic[:,:,run] = Ξhqic
            Ξs_bc[:,:,run] = Ξbc
            Ξs_kic[:,:,run] = Ξkic

        end 
        return data["Ξtrue"], Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
    end
end

function GetInputs(Xs , ts, Lib::Function, NoisePct; p=10, est_der=false)
    if !est_der
        Xtrue = Xs[1]
        η = NoisePct*mean(std(Xtrue, dims=2))/100  
        Xn = Xtrue + η.*randn(size(Xtrue))
        Xsm = smooth_ode(Xn)
        _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)  
        qt = dwInt_sc(Xsm, ts, wind, p=p)
        θ = Lib(Xsm, ts, wind, p=p)

        for i=2:length(Xs)
            Xtrue = Xs[i]
            η = NoisePct*mean(std(Xtrue, dims=2))/100  
            Xn = Xtrue + η.*randn(size(Xtrue))
            Xsm = smooth_ode(Xn)
            _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)  
            qt_i = dwInt_sc(Xsm, ts, wind, p=p)
            θ_i = Lib(Xsm, ts, wind, p=p)

            qt = hcat(qt, qt_i)
            θ = vcat(θ,θ_i)
        end

        return qt, θ
    else # for 1-D systems where we don't know velocity (VdP, NLP)
        xtrue = Xs[1][1,:]
        η = NoisePct*std(xtrue)/100  
        xn = xtrue + η.*randn(size(xtrue))
        xsm = smooth_ode(xn)
        ts_tr, x, dx = DataWithFirstDeriv(xsm', ts, ts[2]-ts[1])
        Xsm = vcat(x, dx)
        _,_,wind = FindW(Xsm, ts_tr, ws=21:2:121, p=p)  
        qt = dwInt_sc(Xsm, ts_tr, wind, p=p)
        θ = Lib(Xsm, ts_tr, wind, p=p)

        for i=2:length(Xs)
            xtrue = Xs[i][1,:]
            η = NoisePct*std(xtrue)/100  
            xn = xtrue + η.*randn(size(xtrue))
            xsm = smooth_ode(xn)
            ts_tr, x, dx = DataWithFirstDeriv(xsm', ts, ts[2]-ts[1])
            Xsm = vcat(x, dx)
            _,_,wind = FindW(Xsm, ts_tr, ws=21:2:121, p=p)  
            qt_i = dwInt_sc(Xsm, ts_tr, wind, p=p)
            θ_i = Lib(Xsm, ts_tr, wind, p=p)

            qt = hcat(qt, qt_i)
            θ = vcat(θ,θ_i)
        end

        return qt, θ
    end
end

function GetInputs(sys::Int, Xs , ts, Lib::Function, NoisePct; p=10)
    if sys==1 || sys==2 || sys==3 || sys==4
        Xtrue = Xs[1]
        η = NoisePct*mean(std(Xtrue, dims=2))/100  
        Xn = Xtrue + η.*randn(size(Xtrue))
        Xsm = smooth_ode(Xn)
        _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)  
        qt = dwInt_sc(Xsm, ts, wind, p=p)
        θ = Lib(Xsm, ts, wind, p=p)

        for i=2:length(Xs)
            Xtrue = Xs[i]
            η = NoisePct*mean(std(Xtrue, dims=2))/100  
            Xn = Xtrue + η.*randn(size(Xtrue))
            Xsm = smooth_ode(Xn)
            _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)  
            qt_i = dwInt_sc(Xsm, ts, wind, p=p)
            θ_i = Lib(Xsm, ts, wind, p=p)

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
        qt = dwInt_sc(Xsm, ts_tr, wind, p=p)
        θ = Lib(Xsm, ts_tr, wind, p=p)

        for i=2:length(Xs)
            xtrue = Xs[i][1,:]
            η = NoisePct*std(xtrue)/100  
            xn = xtrue + η.*randn(size(xtrue))
            xsm = smooth_ode(xn)
            ts_tr, x, dx = DataWithFirstDeriv(xsm', ts, ts[2]-ts[1])
            Xsm = vcat(x, dx)
            _,_,wind = FindW(Xsm, ts_tr, ws=21:2:121, p=p)  
            qt_i = dwInt_sc(Xsm, ts_tr, wind, p=p)
            θ_i = Lib(Xsm, ts_tr, wind, p=p)

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
        qt = d2wInt_sc(Xsm, ts, wind, p=p)
        qt = reshape(qt, (length(qt),1))
        θ = Lib(Xsm, ts, wind, p=p)

        for i=2:length(Xs)
            xtrue = Xs[i][1,:]
            η = NoisePct*std(xtrue)/100  
            xn = xtrue + η.*randn(size(xtrue))
            xsm = smooth_ode(xn)
            Xsm = reshape(xsm, (1, length(xsm)))
            _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p) 
            qt_i = d2wInt_sc(Xsm, ts, wind, p=p)
            qt_i = reshape(qt_i, (length(qt_i),1))
            θ_i = Lib(Xsm, ts, wind, p=p)

            qt = vcat(qt, qt_i)
            θ = vcat(θ,θ_i)
        end

        return qt, θ
    end
end

function DiscoverODEModel(sys::Int, NoisePcts::Vector{Int64}, runs::Int; 
    tol=0.7, 
    c=1e-2, 
    s=1, 
    num_batches=10
    )
    @assert 1 <= sys <= 6 && sort(NoisePcts)[1] >=0

    NoisePct = NoisePcts[1]
    println("sys => $(sys) , NoisePct => $(NoisePct)")    
    Ξtrue, Ξaics, Ξaiccs, Ξslics, Ξbics, Ξhqics, Ξbcs, Ξkics  = DiscoverODEModel(sys, NoisePct, 
    runs, tol=tol, num_batches=num_batches, c=c)

    Ξs_aicc = zeros(size(Ξtrue)..., runs, length(NoisePcts))
    Ξs_slic = similar(Ξs_aicc)
    Ξs_aic = similar(Ξs_aicc)
    Ξs_bic = similar(Ξs_aicc)
    Ξs_hqic = similar(Ξs_aicc)
    Ξs_bc = similar(Ξs_aicc)
    Ξs_kic = similar(Ξs_aicc)

    Ξs_aicc[:,:,:,1] = Ξaiccs
    Ξs_slic[:,:,:,1] = Ξslics
    Ξs_bic[:,:,:,1] = Ξbics
    Ξs_aic[:,:,:,1] = Ξaics
    Ξs_hqic[:,:,:,1] = Ξhqics
    Ξs_bc[:,:,:,1] = Ξbcs
    Ξs_kic[:,:,:,1] = Ξkics

    for i=2:length(NoisePcts)
        
        # add noise
        NoisePct = NoisePcts[i]
        println("sys => $(sys) , NoisePct => $(NoisePct)")    

        Ξtrue, Ξaics, Ξaiccs, Ξslics, Ξbics, Ξhqics, Ξbcs, Ξkics  = DiscoverODEModel(sys, NoisePct, 
        runs, tol=tol, num_batches=num_batches, c=c)

        Ξs_aicc[:,:,:,i] = Ξaiccs
        Ξs_slic[:,:,:,i] = Ξslics
        Ξs_bic[:,:,:,i] = Ξbics
        Ξs_aic[:,:,:,i] = Ξaics
        Ξs_hqic[:,:,:,i] = Ξhqics
        Ξs_bc[:,:,:,i] = Ξbcs
        Ξs_kic[:,:,:,i] = Ξkics
    end
    return Ξtrue, Ξs_aic, Ξs_aicc, Ξs_slic, Ξs_bic, Ξs_hqic, Ξs_bc, Ξs_kic
end

# parameters
NoisePcts = [0, 5, 10, 20, 30, 40]
tol = 0.7
num_batches = 20 
runs = 25 

# Lorenz
sys = 1 
c = 1e-1
Ξtrue_lor, Ξaics_lor, Ξaiccs_lor, Ξslics_lor, Ξbics_lor, Ξhqics_lor, Ξbcs_lor, Ξkics_lor  = DiscoverODEModel(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

data = load(datadir("sims", "ode_data", "lordata.jld"))

lordata = Dict(
    "ts" => data["ts"],
    "Xtrue" => data["Xtrue"],
    "DXtrue" => data["DXtrue"],
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

# Rossler 
sys = 2 
c=1e-1
Ξtrue_ross, Ξaics_ross, Ξaiccs_ross, Ξslics_ross, Ξbics_ross, Ξhqics_ross, Ξbcs_ross, Ξkics_ross  = DiscoverODEModel(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

data = load(datadir("sims", "ode_data", "rossdata.jld"))

rossdata = Dict(
    "ts" => data["ts"],
    "Xtrue" => data["Xtrue"],
    "DXtrue" => data["DXtrue"],
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

# L-V
sys = 3 
c=1e-3
Ξtrue_lv, Ξaics_lv, Ξaiccs_lv, Ξslics_lv, Ξbics_lv, Ξhqics_lv, Ξbcs_lv, Ξkics_lv  = DiscoverODEModel(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

data = load(datadir("sims", "ode_data", "lvdata.jld"))

lvdata = Dict(
    "ts" => data["ts"],
    "Xtrue" => data["Xtrue"],
    "DXtrue" => data["DXtrue"],
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

# Brusselator
sys = 4 
c=1e-2
Ξtrue_brus, Ξaics_brus, Ξaiccs_brus, Ξslics_brus, Ξbics_brus, Ξhqics_brus, Ξbcs_brus, Ξkics_brus  = DiscoverODEModel(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

data = load(datadir("sims", "ode_data", "brusdata.jld"))

brusdata = Dict(
    "ts" => data["ts"],
    "Xtrue" => data["Xtrue"],
    "DXtrue" => data["DXtrue"],
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

# VdP
sys = 5 
c=1e-0
Ξtrue_vdp, Ξaics_vdp, Ξaiccs_vdp, Ξslics_vdp, Ξbics_vdp, Ξhqics_vdp, Ξbcs_vdp, Ξkics_vdp  = DiscoverODEModel(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

data = load(datadir("sims", "ode_data", "vdpdata.jld"))

vdpdata = Dict(
    "ts" => data["ts"],
    "Xtrue" => data["Xtrue"],
    "DXtrue" => data["DXtrue"],
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

# NLP
sys = 6 
c=1e-2
Ξtrue_nlp, Ξaics_nlp, Ξaiccs_nlp, Ξslics_nlp, Ξbics_nlp, Ξhqics_nlp, Ξbcs_nlp, Ξkics_nlp  = DiscoverODEModel(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

data = load(datadir("sims", "ode_data", "nlpdata.jld"))

nlpdata = Dict(
    "ts" => data["ts"],
    "Xtrue" => data["Xtrue"],
    "DXtrue" => data["DXtrue"],
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
 

#= 
wsave(datadir("sims", "ode_results_main", "lor_results_main.jld"), lordata)
wsave(datadir("sims", "ode_results_main", "ross_results_main.jld"), rossdata)
wsave(datadir("sims", "ode_results_main", "lv_results_main.jld"), lvdata)
wsave(datadir("sims", "ode_results_main", "brus_results_main.jld"), brusdata)
wsave(datadir("sims", "ode_results_main", "vdp_results_main.jld"), vdpdata)
wsave(datadir("sims", "ode_results_main", "nlp_results_main.jld"), nlpdata)
=#   