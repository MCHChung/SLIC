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
# Ξs_slics = slic-determined models  
# Ξs_aiccs = aicc-determined models 

function DiscoverODEModel(sys::Int, NoisePct, runs::Int; 
    tol=0.7, 
    c=1e-2, 
    s=1, 
    num_batches=10
    )
    @assert 1 <= sys <= 6 && NoisePct >=0
    if sys == 1 # Lorenz
        # get data 
        data = load(datadir("sims", files[2], "lor_results_main.jld"))
        ts = data["ts"][:]
        Xtrue = data["Xtrue"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
        Ξs_slic = similar(Ξs_aicc)
        for run=1:runs
            # add noise
            η = NoisePct*std(Xtrue)/100  
            Xn = Xtrue + η.*randn(size(Xtrue))
            Xsm = smooth_ode(Xn)
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
            # set polynomial order and determine SSIM-opt window 
            p = 10
            _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)  

            # peform galerkin-projected 
            qt = dwInt_sc(Xsm, ts, wind, p=p)
            θ = LorLib(Xsm, ts, wind, p=p, Δ=s)

            Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
            Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)

            Ξs_aicc[:,:,run] = Ξaicc
            Ξs_slic[:,:,run] = Ξslic
        end
        return data["Ξtrue"], Ξs_aicc, Ξs_slic
    elseif sys == 2 # Rossler
        # get data 
        data = load(datadir("sims", files[2], "ross_results_main.jld"))
        ts = data["ts"][:]
        Xtrue = data["Xtrue"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
        Ξs_slic = similar(Ξs_aicc)
        # add noise
        for run=1:runs
            η = NoisePct*std(Xtrue)/100  
            Xn = Xtrue + η.*randn(size(Xtrue))
            Xsm = smooth_ode(Xn)
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
            # set polynomial order and determine SSIM-opt window 
            p = 10
            _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)  

            # peform galerkin-projected 
            qt = dwInt_sc(Xsm, ts, wind, p=p)
            θ = RossLib(Xsm, ts, wind, p=p, Δ=s)

            Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
            Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)

            Ξs_aicc[:,:,run] = Ξaicc
            Ξs_slic[:,:,run] = Ξslic
        end
        return data["Ξtrue"], Ξs_aicc, Ξs_slic
    elseif sys == 3 # Lotka-Volterra
        # get data 
        data = load(datadir("sims", files[2], "lv_results_main.jld"))
        ts = data["ts"][:]
        Xtrue = data["Xtrue"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
        Ξs_slic = similar(Ξs_aicc)
        for run=1:runs
            # add noise
            η = NoisePct*std(Xtrue)/100  
            Xn = Xtrue + η.*randn(size(Xtrue))
            Xsm = smooth_ode(Xn)
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
            # set polynomial order and determine SSIM-opt window 
            p = 10
            _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)  

            # peform galerkin-projected 
            qt = dwInt_sc(Xsm, ts, wind, p=p)
            θ = LVLib(Xsm, ts, wind, p=p, Δ=s)

            Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
            Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)

            Ξs_aicc[:,:,run] = Ξaicc
            Ξs_slic[:,:,run] = Ξslic
        end
        return data["Ξtrue"], Ξs_aicc, Ξs_slic
    elseif sys == 4 # Brusselator
        # get data 
        data = load(datadir("sims", files[2], "brus_results_main.jld"))
        ts = data["ts"][:]
        Xtrue = data["Xtrue"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
        Ξs_slic = similar(Ξs_aicc)
        for run=1:runs
            # add noise
            η = NoisePct*std(Xtrue)/100  
            Xn = Xtrue + η.*randn(size(Xtrue))
            Xsm = smooth_ode(Xn)
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
            # set polynomial order and determine SSIM-opt window 
            p = 10
            _,_,wind = FindW(Xsm, ts, ws=21:2:121, p=p)  

            # peform galerkin-projected 
            qt = dwInt_sc(Xsm, ts, wind, p=p)
            θ = BrusLib(Xsm, ts, wind, p=p, Δ=s)

            Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
            Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)

            Ξs_aicc[:,:,run] = Ξaicc
            Ξs_slic[:,:,run] = Ξslic
        end
        return data["Ξtrue"], Ξs_aicc, Ξs_slic
    elseif sys == 5 # Van der Pol
        # get data 
        data = load(datadir("sims", files[2], "vdp_results_main.jld"))
        ts = data["ts"][:]
        xtrue = data["Xtrue"][1,:] # get x-coordinate
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
        Ξs_slic = similar(Ξs_aicc)
        for run=1:runs
            # add noise
            η = NoisePct*std(xtrue)/100  
            xn = xtrue + η.*randn(size(xtrue))
            xsm = smooth_ode(xn)
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
             
            # estimate 1st deriv  
            ts_tr, x, dx = DataWithFirstDeriv(xsm', ts, ts[2]-ts[1])
            Xsm = vcat(x, dx)
            # set polynomial order and determine SSIM-opt window 
            p = 10
            _,_,wind = FindW(Xsm, ts_tr, ws=21:2:121, p=p)  

            # peform galerkin-projection
            qt = dwInt_sc(Xsm, ts_tr, wind, p=p)
            θ = VdPLib(Xsm, ts_tr, wind, p=p, Δ=s)

            Ξslic, _ = EnAdSR(θ, qt', "slic", tol=tol, c=c, num_batches=num_batches)
            Ξaicc, _ = EnAdSR(θ, qt', "aicc", tol=tol, c=c, num_batches=num_batches)

            Ξs_aicc[:,:,run] = Ξaicc
            Ξs_slic[:,:,run] = Ξslic
        end 
        return data["Ξtrue"], Ξs_aicc, Ξs_slic
    elseif sys == 6 # Nonlinear Pendulum
        # get data 
        data = load(datadir("sims", files[2], "nlp_results_main.jld"))
        ts = data["ts"][:]
        xtrue = data["Xtrue"][1,:]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
        Ξs_slic = similar(Ξs_aicc)
        for run=1:runs
            # add noise
            η = NoisePct*std(xtrue)/100  
            xn = xtrue + η.*randn(size(xtrue))
            xsm = smooth_ode(xn)
            xsm = reshape(xsm, (1, length(xsm)))
            # define lib 
            function NLPLib(X, ts, wind; p=10, Δ=1)
                θ = wInt_sc(ones(length(X[:])), ts, wind, p=p, Δ=Δ)
                for i=1:5
                    θ = hcat(θ, wInt_sc(sin.(i*X[:]), ts, wind, p=p, Δ=Δ))
                    θ = hcat(θ, wInt_sc(cos.(i*X[:]), ts, wind, p=p, Δ=Δ))
                end
                return θ
            end
            # set polynomial order and determine SSIM-opt window 
            p = 10
            _,_,wind = FindW(xsm, ts, ws=21:2:121, p=p)  

            # peform galerkin-projection, solving d2x/dt2 = f(x)
            qt = d2wInt_sc(xsm, ts, wind, p=p) # second order deriv! 
            qt = reshape(qt, (length(qt),1))
            θ = NLPLib(xsm, ts, wind, p=p, Δ=s)

            Ξslic, _ = EnAdSR(θ, qt, "slic", tol=tol, c=c, num_batches=num_batches)
            Ξaicc, _ = EnAdSR(θ, qt, "aicc", tol=tol, c=c, num_batches=num_batches)

            Ξs_aicc[:,:,run] = Ξaicc
            Ξs_slic[:,:,run] = Ξslic
        end 
        return data["Ξtrue"], Ξs_aicc, Ξs_slic
    end
end

sys = 3 
NoisePct = 20
runs = 5 
Ξtrue, Ξaiccs, Ξslics = DiscoverODEModel(sys, NoisePct, runs, tol=0.7, num_batches=10)

# visualize the mean of the models across the different noise instantiations  
paic = VisResults(Ξtrue, mean(Ξaiccs, dims=3)[:,:], "AICc")
pslic = VisResults(Ξtrue, mean(Ξslics, dims=3)[:,:], "SLIC")

display(paic)
display(pslic)