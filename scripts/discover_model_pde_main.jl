using DrWatson
@quickactivate "SLIC"
using JLD

# functions to extract models and visualize results
include(srcdir("sparse_regress.jl"))
include(srcdir("galerkin_proj.jl"))
include(srcdir("smooth.jl"))
include(srcdir("vis_results.jl"))

# all the pde data 
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

function DiscoverPDEModel(sys::Int, NoisePct::Int64, runs::Int; 
    tol=0.7, 
    c=1e-7, 
    s=1, 
    num_batches=10
    )
    @assert 1 <= sys <= 5 && NoisePct >= 0
    if sys == 1 # burger
        # get data 
        data = load(datadir("sims", "pde_data", "bgdata.jld"))
        tmat = data["ts"][:]
        xmat = data["xs"][:]
        doms = (tmat, xmat)
        umat = data["Xtrue"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)

        for run=1:runs
            # add noise
            println("NoisePct => $(NoisePct), run => $(run)")
            η = NoisePct*std(umat)/100  
            un = umat + η.*randn(size(umat))
            usm = smooth_pde(un)
            # define lib 
            function BurgLib(u, doms, ws , ps, s)
                # constant terms
                f1 = vec(IntData(ones(size(u)), doms, (0,0) , ws, ps, s))
                # linear in u 
                f2 = vec(IntData(u, doms, (0,0), ws, ps, s))
                # first x deriv 
                f3 = vec(IntData(u, doms, (0,1), ws, ps,s))
                # second x deriv 
                f4 = vec(IntData(u, doms, (0,2), ws, ps, s))
                # second x deriv 
                f5 = vec(IntData(u, doms, (0,3) , ws, ps, s))
                # 4th deriv
                f6 = vec(IntData(u, doms, (0,4), ws ,ps, s))
                # quadratic in u 
                f7 = vec(IntData(u.^2, doms, (0,0) ,ws, ps, s))
                # convective u*dxu 
                f8 = vec(IntData(0.5*u.^2, doms, (0,1) ,ws, ps, s))
            
                f9 = vec(IntData(u.^3, doms, (0,0), ws, ps, s))
            
                return hcat(f1,f2,f3,f4,f5,f6,f7,f8,f9)
            end
            # set polynomial order and determine SSIM-opt window 
            ps = (8,8)
            _,_,wind = FindW(usm, tmat, xmat, ps, ws=25:2:55)  

            # peform galerkin-projection 
            qt = vec(IntData(usm, doms, (1,0), wind, ps,s))
            qt = reshape(qt, (length(qt),1))
            θ = BurgLib(usm, doms, wind, ps,s)

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
    elseif sys == 2 # kdv
        # get data 
        data = load(datadir("sims", "pde_data", "kdvdata.jld"))
        tmat = data["ts"][:]
        xmat = data["xs"][:]
        doms = (tmat, xmat)
        umat = data["Xtrue"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)

        for run=1:runs
            println("NoisePct => $(NoisePct), run => $(run)")
            # add noise
            η = NoisePct*std(umat)/100  
            un = umat + η.*randn(size(umat))
            usm = smooth_pde(un)
            # define lib 
            function KdVLib(u, doms, ws, ps, s)
                # constant terms
                f1 = vec(IntData(ones(size(u)), doms, (0,0), ws, ps,s))
                # linear in u 
                f2 = vec(IntData(u, doms, (0,0), ws, ps,s))
                # first x deriv 
                f3 = vec(IntData(u, doms, (0,1), ws, ps,s))
                # second x deriv 
                f4 = vec(IntData(u, doms, (0,2), ws, ps,s))
                # 3rd x deriv 
                f5 = vec(IntData(u, doms, (0,3), ws, ps,s))
                # 4th deriv
                f6 = vec(IntData(u, doms, (0,4), ws, ps,s))
                # quadratic in u 
                f7 = vec(IntData(u.^2, doms, (0,0), ws, ps,s))
                # convective u*dxu 
                f8 = vec(IntData(0.5*u.^2, doms, (0,1), ws, ps,s))
                # cubic
                f9 = vec(IntData(u.^3, doms, (0,0), ws, ps,s))
            
                return hcat(f1,f2,f3,f4,f5,f6,f7,f8,f9)
            end
            # set polynomial order and determine SSIM-opt window 
            ps = (8,8)
            _,_,wind = FindW(usm, tmat, xmat, ps, ws=25:2:55)  

            # peform galerkin-projection 
            qt = vec(IntData(usm, doms, (1,0), wind, ps,s))
            qt = reshape(qt, (length(qt),1))
            θ = KdVLib(usm, doms, wind, ps,s)

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
    elseif sys == 3 # K-S
        # get data 
        data = load(datadir("sims", "pde_data", "ksdata.jld"))
        tmat = data["ts"][:]
        xmat = data["xs"][:]
        doms = (tmat, xmat)
        umat = data["Xtrue"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)

        for run=1:runs
            println("NoisePct => $(NoisePct), run => $(run)")
            # add noise
            η = NoisePct*std(umat)/100  
            un = umat + η.*randn(size(umat))
            usm = smooth_pde(un)
            # define lib 
            function KSLib(u, doms, ws, ps, s)
                # constant terms
                f1 = vec(IntData(ones(size(u)), doms, (0,0),  ws, ps,s))
                # linear in u 
                f2 = vec(IntData(u, doms, (0,0),  ws, ps,s))
                # first x deriv 
                f3 = vec(IntData(u, doms, (0,1),  ws, ps,s))
                # second x deriv 
                f4 = vec(IntData(u, doms, (0,2),  ws, ps,s))
                # 3rd x deriv 
                f5 = vec(IntData(u, doms, (0,3),  ws, ps,s))
                # 4th deriv
                f6 = vec(IntData(u, doms, (0,4),  ws, ps,s))
                # quadratic in u 
                f7 = vec(IntData(u.^2, doms, (0,0),  ws, ps,s))
                # convective u*dxu 
                f8 = vec(IntData(0.5*u.^2, doms, (0,1),  ws, ps,s))
                # cubic
                f9 = vec(IntData(u.^3, doms, (0,0),  ws, ps,s))
            
                return hcat(f1,f2,f3,f4,f5,f6,f7,f8,f9)
            end
            # set polynomial order and determine SSIM-opt window 
            ps = (8,8)
            _,_,wind = FindW(usm, tmat, xmat, ps, ws=25:2:55)  

            # peform galerkin-projection 
            qt = vec(IntData(usm, doms, (1,0), wind, ps,s))
            qt = reshape(qt, (length(qt),1))
            θ = KSLib(usm, doms, wind, ps,s)

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
    elseif sys == 4 # NLS
        # get data 
        data = load(datadir("sims", "pde_data", "nlsdata.jld"))
        tmat = data["ts"][:]
        xmat = data["xs"][:]
        doms = (tmat, xmat)
        umat = data["Xtrue"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)

        for run=1:runs
            println("NoisePct => $(NoisePct), run => $(run)")
            # add noise
            η = NoisePct*std(umat)/100  
            un = umat + η.*randn(size(umat))
            usm = smooth_pde(un[:,:,1])
            vsm = smooth_pde(un[:,:,2])
            # define lib 
            function NLSLib(u,v, doms, ws, ps, s)
                # constant terms
                f1 = vec(IntData(ones(size(u)), doms, (0,0), ws, ps, s))
                # linear in u 
                f2 = vec(IntData(u, doms, (0,0), ws, ps, s))
                # linear in v 
                f3 = vec(IntData(v, doms, (0,0), ws, ps, s))
                # first x deriv in u
                f4 = vec(IntData(u, doms, (0,1), ws, ps, s))
                # first x deriv in v
                f5 = vec(IntData(v, doms, (0,1), ws, ps, s))
                # second x deriv in u
                f6 = vec(IntData(u, doms, (0,2), ws, ps, s))
                # second x deriv in v
                f7 = vec(IntData(v, doms, (0,2), ws, ps, s))
                # quadratic in u 
                f8 = vec(IntData(u.^2, doms, (0,3), ws, ps, s))
                # quadratic in v 
                f9 = vec(IntData(v.^2, doms, (0,3), ws, ps, s))
                # convective u*dxu 
                f8 = vec(IntData(0.5*u.^2, doms, (0,1), ws, ps, s))
                # convective v*dxv 
                f8 = vec(IntData(0.5*v.^2, doms, (0,1), ws, ps, s))
                # quadratic in u*v 
                f9 = vec(IntData(u.*v, doms, (0,0), ws, ps, s))
                # cubi in u 
                f10 = vec(IntData(u.^3, doms, (0,0), ws, ps, s))
                # cubic in v 
                f11 = vec(IntData(v.^3, doms, (0,0), ws, ps, s))
                # quadratic in u*v 
                f12 = vec(IntData(u.*v.^2, doms, (0,0), ws, ps, s))
                # quadratic in u*v 
                f13 = vec(IntData(u.^2 .*v, doms, (0,0), ws, ps, s))
            
                return hcat(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13)
            end
            # set polynomial order and determine SSIM-opt window 
            ps = (8,8)
            _,_,wind = FindW(usm, tmat, xmat, ps, ws=25:2:55)  

            # peform galerkin-projection 
            qtu = vec(IntData(usm, doms, (1,0), wind, ps, s))
            qtv = vec(IntData(vsm, doms, (1,0), wind, ps, s))
            qt = vcat(qtu', qtv')
            θ = NLSLib(usm, vsm, doms, wind, ps, s)

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
    elseif sys == 5 # SG
        # get data 
        data = load(datadir("sims", "pde_data", "sgdata.jld"))
        tmat = data["ts"][:]
        xmat = data["xs"][:]
        ymat = data["ys"][:]
        doms = (tmat, xmat, ymat)
        umat = data["Xtrue"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
        Ξs_slic = similar(Ξs_aicc)
        Ξs_aic = similar(Ξs_aicc)
        Ξs_bic = similar(Ξs_aicc)
        Ξs_hqic = similar(Ξs_aicc)
        Ξs_bc = similar(Ξs_aicc)
        Ξs_kic = similar(Ξs_aicc)

        for run=1:runs
            println("NoisePct => $(NoisePct), run => $(run)")
            # add noise
            η = NoisePct*std(umat)/100  
            un = umat + η.*randn(size(umat))
            usm = smooth_pde(un)
            # define lib 
            function SGLib(u, doms, ws, ps, s)
                #X = vec(u)
                pt,px,py = ps
                #ws = wind*[1,1,1]
                f1 = vec(wInt_sc(ones(size(u)), doms... , ws , pt=pt,px=px,py=py, Δ=s))
                f2 = vec(dtwInt_sc(u, doms... ,  ws  , pt=pt,px=px,py=py, Δ=s))
                f3 = vec(dxwInt_sc(u, doms... ,  ws  , pt=pt,px=px,py=py, Δ=s))
                f4 = vec(dywInt_sc(u, doms... ,  ws  , pt=pt,px=px,py=py, Δ=s))
                f5 = vec(d2xwInt_sc(u, doms... ,  ws  , pt=pt,px=px,py=py, Δ=s))
                f6 = vec(d2ywInt_sc(u, doms... ,  ws  , pt=pt,px=px,py=py, Δ=s))
                f7 = vec(dxywInt_sc(u, doms... , ws  , pt=pt,px=px,py=py, Δ=s))
                f8 = vec(wInt_sc(u , doms... ,  ws  , pt=pt,px=px,py=py, Δ=s))
                f9 = vec(wInt_sc(u.^2, doms... ,  ws  , pt=pt,px=px,py=py, Δ=s))
                f10 = vec(wInt_sc(sin.(u), doms... ,  ws  , pt=pt,px=px,py=py, Δ=s))
                f11 = vec(wInt_sc(cos.(u), doms... ,  ws  , pt=pt,px=px,py=py, Δ=s))

                return [f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11]
            end
            # set polynomial order and determine SSIM-opt window 
            ps = [8,8,8]
            #_,_,wind = FindW(usm[:,:,1], tmat, xmat, ps, ws=25:2:55)  
            wind = 55
            # peform galerkin-projection 
            qt = vec(d2twInt_sc(usm, doms..., wind*[1,1,1], pt=8,px=8,py=8, Δ=s))
            qt = reshape(qt, (length(qt), 1))
            θ = SGLib(usm, doms , wind*[1,1,1] , ps, s)

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

function DiscoverPDEModel(sys::Int, NoisePcts::Vector{Int64}, runs::Int; 
    tol=0.7, 
    c=1e-7, 
    s=1, 
    num_batches=10
    )
    @assert 1 <= sys <= 6 && sort(NoisePcts)[1] >=0

    NoisePct = NoisePcts[1]
    println("sys => $(sys) , NoisePct => $(NoisePct)")    
    Ξtrue, Ξaics, Ξaiccs, Ξslics, Ξbics, Ξhqics, Ξbcs, Ξkics  = DiscoverPDEModel(sys, NoisePct, 
    runs, tol=tol, num_batches=num_batches, c=c, s=s)

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

        Ξtrue, Ξaics, Ξaiccs, Ξslics, Ξbics, Ξhqics, Ξbcs, Ξkics  = DiscoverPDEModel(sys, NoisePct, 
        runs, tol=tol, num_batches=num_batches, c=c, s=s)

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

# Burger
sys = 1 
c = 1e-7
print("sys => $(sys)")
Ξtrue_bg, Ξaics_bg, Ξaiccs_bg, Ξslics_bg, Ξbics_bg, Ξhqics_bg, Ξbcs_bg, Ξkics_bg  = DiscoverPDEModel(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

data = load(datadir("sims", "pde_data", "bgdata.jld"))

bgdata = Dict(
    "ts" => data["ts"],
    "xs" => data["xs"],
    "Xtrue" => data["Xtrue"],
    "Ξtrue" => Ξtrue_bg, 
    "Ξaics" => Ξaics_bg,
    "Ξaiccs" => Ξaiccs_bg, 
    "Ξbics" => Ξbics_bg,
    "Ξhqics" => Ξhqics_bg,
    "Ξkics" => Ξkics_bg,
    "Ξbcs" => Ξbcs_bg,
    "Ξslics" => Ξslics_bg,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)


# KdV 
sys = 2 
c=1e-7
print("sys => $(sys)")
Ξtrue_kdv, Ξaics_kdv, Ξaiccs_kdv, Ξslics_kdv, Ξbics_kdv, Ξhqics_kdv, Ξbcs_kdv, Ξkics_kdv  = DiscoverPDEModel(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

data = load(datadir("sims", "pde_data", "kdvdata.jld"))

kdvdata = Dict(
    "ts" => data["ts"],
    "xs" => data["xs"],
    "Xtrue" => data["Xtrue"],
    "Ξtrue" => Ξtrue_kdv, 
    "Ξaics" => Ξaics_kdv,
    "Ξaiccs" => Ξaiccs_kdv, 
    "Ξbics" => Ξbics_kdv,
    "Ξhqics" => Ξhqics_kdv,
    "Ξkics" => Ξkics_kdv,
    "Ξbcs" => Ξbcs_kdv,
    "Ξslics" => Ξslics_kdv,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)


# KS
sys = 3 
c=1e-4
print("sys => $(sys)")
Ξtrue_ks, Ξaics_ks, Ξaiccs_ks, Ξslics_ks, Ξbics_ks, Ξhqics_ks, Ξbcs_ks, Ξkics_ks  = DiscoverPDEModel(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

data = load(datadir("sims", "pde_data", "ksdata.jld"))

ksdata = Dict(
    "ts" => data["ts"],
    "xs" => data["xs"],
    "Xtrue" => data["Xtrue"],
    "Ξtrue" => Ξtrue_ks, 
    "Ξaics" => Ξaics_ks,
    "Ξaiccs" => Ξaiccs_ks, 
    "Ξbics" => Ξbics_ks,
    "Ξhqics" => Ξhqics_ks,
    "Ξkics" => Ξkics_ks,
    "Ξbcs" => Ξbcs_ks,
    "Ξslics" => Ξslics_ks,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)


# NLS
sys = 4 
c=1e-7
print("sys => $(sys)")
Ξtrue_nls, Ξaics_nls, Ξaiccs_nls, Ξslics_nls, Ξbics_nls, Ξhqics_nls, Ξbcs_nls, Ξkics_nls  = DiscoverPDEModel(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c)

data = load(datadir("sims", "pde_data", "nlsdata.jld"))

nlsdata = Dict(
    "ts" => data["ts"],
    "xs" => data["xs"],
    "Xtrue" => data["Xtrue"],
    "Ξtrue" => Ξtrue_nls, 
    "Ξaics" => Ξaics_nls,
    "Ξaiccs" => Ξaiccs_nls, 
    "Ξbics" => Ξbics_nls,
    "Ξhqics" => Ξhqics_nls,
    "Ξkics" => Ξkics_nls,
    "Ξbcs" => Ξbcs_nls,
    "Ξslics" => Ξslics_nls,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)


# SG
sys = 5 
c=1e-7
print("sys => $(sys)")
Ξtrue_sg, Ξaics_sg, Ξaiccs_sg, Ξslics_sg, Ξbics_sg, Ξhqics_sg, Ξbcs_sg, Ξkics_sg  = DiscoverPDEModel(sys, 
NoisePcts, runs, tol=tol, num_batches=num_batches, c=c, s=5)

data = load(datadir("sims", "pde_data", "sgdata.jld"))

sgdata = Dict(
    "ts" => data["ts"],
    "xs" => data["xs"],
    "ys" => data["ys"],
    "Xtrue" => data["Xtrue"],
    "Ξtrue" => Ξtrue_sg, 
    "Ξaics" => Ξaics_sg,
    "Ξaiccs" => Ξaiccs_sg, 
    "Ξbics" => Ξbics_sg,
    "Ξhqics" => Ξhqics_sg,
    "Ξkics" => Ξkics_sg,
    "Ξbcs" => Ξbcs_sg,
    "Ξslics" => Ξslics_sg,
    "NoisePcts" => NoisePcts, 
    "tol" => tol, 
    "c" => c, 
    "num_batches" => num_batches 
)
 

#=
# save data 
wsave(datadir("sims", "pde_results_main", "burger_results_main.jld"), bgdata)
wsave(datadir("sims", "pde_results_main", "kdv_results_main.jld"), kdvdata)
wsave(datadir("sims", "pde_results_main", "ks_results_main.jld"), ksdata)
wsave(datadir("sims", "pde_results_main", "nls_results_main.jld"), nlsdata)
wsave(datadir("sims", "pde_results_main", "sg_results_main.jld"), sgdata)
=# 
