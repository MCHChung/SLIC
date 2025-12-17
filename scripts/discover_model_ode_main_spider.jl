using DrWatson, JLD, StatsBase
@quickactivate "SLIC"
using JLD

# functions to extract models and visualize results
include(srcdir("spider_2.jl"))
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
    c=0., 
    s=1, 
    num_batches=10,
    gamma=1.4
    )
    @assert 1 <= sys <= 6 && NoisePct >=0
    if sys == 1 # Lorenz
        # get data 
        data = load(datadir("sims", "ode_data", "lordata.jld"))
        
        # arrays to hold model outputs
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs)
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1, size(data["Ξtrue"],2), runs)

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
            println("run => $(run)")
            # get inputs
            qt, θ =  GetInputs(sys, data["Xtrues"], data["ts"], LorLib, NoisePct)
            
            Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)            
            Ξs_sp[:,:,run] = Ξsp
            Ξs_sp_sc[:,:,run] = Ξsp_sc

        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 2 # Rossler
        # get data 
        data = load(datadir("sims", "ode_data", "rossdata.jld"))
        
        # arrays to hold model outputs
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1, size(data["Ξtrue"],2), runs)
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs)
        

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
            println("run => $(run)")
            qt, θ = GetInputs(sys, data["Xtrues"], data["ts"], RossLib, NoisePct)

            Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)
            Ξs_sp[:,:,run] = Ξsp
            Ξs_sp_sc[:,:,run] = Ξsp_sc
        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 3 # Lotka-Volterra
        # get data 
        data = load(datadir("sims", "ode_data", "lvdata.jld"))
        
        # arrays to hold model outputs
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1, size(data["Ξtrue"],2), runs)
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs)

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
            println("run => $(run)")
            # get inputs 
            qt, θ = GetInputs(sys, data["Xtrues"], data["ts"], LVLib, NoisePct)
            
            # determine dynamics
            Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)
            Ξs_sp[:,:,run] = Ξsp
            Ξs_sp_sc[:,:,run] = Ξsp_sc

        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 4 # Brusselator
        # get data 
        data = load(datadir("sims", "ode_data", "brusdata.jld"))
        
        # arrays to hold model outputs
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1, size(data["Ξtrue"],2), runs)
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs)

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
            println("run => $(run)")
            # get inputs
            qt, θ = GetInputs(sys, data["Xtrues"], data["ts"], BrusLib, NoisePct)
            
            # determine dynamics
            Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)
            Ξs_sp[:,:,run] = Ξsp
            Ξs_sp_sc[:,:,run] = Ξsp_sc

        end
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 5 # Van der Pol
        # get data 
        data = load(datadir("sims", "ode_data", "vdpdata.jld"))
        
        # arrays to hold model outputs
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1, size(data["Ξtrue"],2), runs)
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs)

        
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
            println("run => $(run)")
            # get inputs
            qt, θ = GetInputs(sys, data["Xtrues"], data["ts"], VdPLib, NoisePct)
            
            # discover model
            Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt)
            Ξs_sp[:,:,run] = Ξsp
            Ξs_sp_sc[:,:,run] = Ξsp_sc
            

        end 
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
    elseif sys == 6 # Nonlinear Pendulum
        # get data 
        data = load(datadir("sims", "ode_data", "nlpdata.jld"))
        
        # arrays to hold model outputs
        Ξs_sp = zeros(size(data["Ξtrue"],1)+1, size(data["Ξtrue"],2), runs)
        Ξs_sp_sc = zeros(size(data["Ξtrue"])..., runs)
        

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
            println("run => $(run)")
            # add noise
            qt, θ = GetInputs(sys, data["Xtrues"] , data["ts"], NLPLib, NoisePct; p=10)
            
            # discover model
            Ξsp, Ξsp_sc = Sp_SparseReg_2(θ, qt')
            Ξs_sp[:,:,run] = Ξsp
            Ξs_sp_sc[:,:,run] = Ξsp_sc
           

        end 
        return data["Ξtrue"], Ξs_sp, Ξs_sp_sc
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
    c=0., 
    s=1, 
    num_batches=10,
    gamma=1.4
    )
    @assert 1 <= sys <= 6 && sort(NoisePcts)[1] >=0

    NoisePct = NoisePcts[1]
    println("sys => $(sys) , NoisePct => $(NoisePct)")    
    Ξtrue, Ξsps, Ξsps_sc  = DiscoverODEModel(sys, NoisePct, runs, tol=tol, num_batches=num_batches, c=c, gamma=gamma)

    Ξs_sp_sc = zeros(size(Ξtrue)..., runs, length(NoisePcts))
    Ξs_sp = zeros(size(Ξtrue,1)+1,size(Ξtrue,2), runs, length(NoisePcts))

    Ξs_sp[:,:,:,1] = Ξsps
    Ξs_sp_sc[:,:,:,1] = Ξsps_sc


    for i=2:length(NoisePcts)
        
        # add noise
        NoisePct = NoisePcts[i]
        println("sys => $(sys) , NoisePct => $(NoisePct)")    

        Ξtrue, Ξsps, Ξsps_sc = DiscoverODEModel(sys, NoisePct, runs, tol=tol, num_batches=num_batches, c=c, gamma=gamma)

        Ξs_sp[:,:,:,i] = Ξsps
        Ξs_sp_sc[:,:,:,i] = Ξsps_sc

    end
    return Ξtrue, Ξs_sp, Ξs_sp_sc
end

function DiscoverODEModel(sys::Int, NoisePcts::Vector{Int64}, runs::Int, gammas::Vector; 
    tol=0.7, 
    c=0., 
    s=1, 
    num_batches=10
    )
    @assert 1 <= sys <= 6 && sort(NoisePcts)[1] >=0
    
    Ξs = []
    Ξtrues = []

    for gamma in gammas
        println("gamma => $(gamma)")
        # add noise
        Ξtrue, Ξsps = DiscoverODEModel(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c, gamma=gamma)

        push!(Ξs, Ξsps) 

        if isempty(Ξtrues)
            push!(Ξtrues, Ξtrue)
        end
    end
    return Ξtrues, Ξs
end

# parameters
NoisePcts = [0, 5, 10, 20, 30, 40]
tol = 0.7
num_batches = 20 
runs = 25 

# Lorenz
sys = 1 
c = 0
Ξtrue_lor, Ξsp_lor, Ξsp_sc_lor = DiscoverODEModel(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

data = load(datadir("sims", "ode_data", "lordata.jld"))

lordata = Dict(
    "Ξtrue" => Ξtrue_lor, 
    "Ξsp" => Ξsp_lor,
    "Ξsp_sc" => Ξsp_sc_lor,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)
#VisResults(mean(abs.(Ξsp_lor[4][:,:,:,6]) .!= 0., dims=3)[:,:], "")
# Rossler 
sys = 2 
c=0
Ξtrue_ross, Ξsp_ross , Ξsp_sc_ross = DiscoverODEModel(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

data = load(datadir("sims", "ode_data", "rossdata.jld"))

rossdata = Dict(
    "Ξtrue" => Ξtrue_ross, 
    "Ξsp" => Ξsp_ross,
    "Ξsp_sc" => Ξsp_sc_ross,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# L-V
sys = 3 
c=1e-3
Ξtrue_lv, Ξsp_lv, Ξsp_sc_lv  = DiscoverODEModel(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

data = load(datadir("sims", "ode_data", "lvdata.jld"))

lvdata = Dict(
    "Ξtrue" => Ξtrue_lv, 
    "Ξsp" => Ξsp_lv,
    "Ξsp_sc" => Ξsp_sc_lv,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# Brusselator
sys = 4 
c=1e-2
Ξtrue_brus, Ξsp_brus, Ξsp_sc_brus  = DiscoverODEModel(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

data = load(datadir("sims", "ode_data", "brusdata.jld"))

brusdata = Dict(
    "Ξtrue" => Ξtrue_brus, 
    "Ξsp" => Ξsp_brus,
    "Ξsp_sc" => Ξsp_sc_brus,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# VdP
sys = 5 
c=1e-0
Ξtrue_vdp, Ξsp_vdp, Ξsp_sc_vdp  = DiscoverODEModel(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

data = load(datadir("sims", "ode_data", "vdpdata.jld"))

vdpdata = Dict(
    "Ξtrue" => Ξtrue_vdp, 
    "Ξsp" => Ξsp_vdp,
    "Ξsp_sc" => Ξsp_sc_vdp,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)

# NLP
sys = 6 
c=1e-2
Ξtrue_nlp, Ξsp_nlp, Ξsp_sc_nlp  = DiscoverODEModel(sys, NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

data = load(datadir("sims", "ode_data", "nlpdata.jld"))

nlpdata = Dict(
    "Ξtrue" => Ξtrue_nlp, 
    "Ξsp" => Ξsp_nlp,
    "Ξsp_sc" => Ξsp_sc_nlp,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)
 

#=
wsave(datadir("sims", "ode_results_main", "lor_spider_results_main.jld"), lordata)
wsave(datadir("sims", "ode_results_main", "ross_spider_results_main.jld"), rossdata)
wsave(datadir("sims", "ode_results_main", "lv_spider_results_main.jld"), lvdata)
wsave(datadir("sims", "ode_results_main", "brus_spider_results_main.jld"), brusdata)
wsave(datadir("sims", "ode_results_main", "vdp_spider_results_main.jld"), vdpdata)
wsave(datadir("sims", "ode_results_main", "nlp_spider_results_main.jld"), nlpdata)
=#  