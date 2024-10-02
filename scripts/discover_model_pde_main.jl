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
# Ξs_slics = slic-determined models  
# Ξs_aiccs = aicc-determined models 

function DiscoverPDEModel(sys::Int, NoisePct, runs::Int; 
    tol=0.7, 
    c=1e-7, 
    s=1, 
    num_batches=10
    )
    @assert 1 <= sys <= 5 && NoisePct >= 0
    if sys == 1 # burger
        # get data 
        data = load(datadir("sims", files[5], "burger_results_main.jld"))
        tmat = data["ts"][:]
        xmat = data["xs"][:]
        doms = (tmat, xmat)
        umat = data["Xtrue"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
        Ξs_slic = similar(Ξs_aicc)
        for run=1:runs
            # add noise
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

            Ξs_aicc[:,:,run] = Ξaicc 
            Ξs_slic[:,:,run] = Ξslic
        end
        return data["Ξtrue"], Ξs_aicc, Ξs_slic
    elseif sys == 2 # kdv
        # get data 
        data = load(datadir("sims", files[5], "kdv_results_main.jld"))
        tmat = data["ts"][:]
        xmat = data["xs"][:]
        doms = (tmat, xmat)
        umat = data["Xtrue"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
        Ξs_slic = similar(Ξs_aicc)
        for run=1:runs
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

            Ξs_aicc[:,:,run] = Ξaicc
            Ξs_slic[:,:,run] = Ξslic
        end 
        return data["Ξtrue"], Ξs_aicc, Ξs_slic
    elseif sys == 3 # K-S
        # get data 
        data = load(datadir("sims", files[5], "ks_results_main.jld"))
        tmat = data["ts"][:]
        xmat = data["xs"][:]
        doms = (tmat, xmat)
        umat = data["Xtrue"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
        Ξs_slic = similar(Ξs_aicc)
        for run=1:runs
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

            Ξs_aicc[:,:,run] = Ξaicc
            Ξs_slic[:,:,run] = Ξslic
        end
        return data["Ξtrue"], Ξs_aicc, Ξs_slic
    elseif sys == 4 # NLS
        # get data 
        data = load(datadir("sims", files[5], "nls_results_main.jld"))
        tmat = data["ts"][:]
        xmat = data["xs"][:]
        doms = (tmat, xmat)
        umat = data["Xtrue"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
        Ξs_slic = similar(Ξs_aicc)
        for run=1:runs
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

            Ξs_aicc[:,:,run] = Ξaicc
            Ξs_slic[:,:,run] = Ξslic
        end 
        return data["Ξtrue"], Ξs_aicc, Ξs_slic
    elseif sys == 5 # SG
        # get data 
        data = load(datadir("sims", files[5], "sg_results_main.jld"))
        tmat = data["ts"][:]
        xmat = data["xs"][:]
        ymat = data["ys"][:]
        doms = (tmat, xmat, ymat)
        umat = data["Xtrue"]
        # arrays to hold model outputs
        Ξs_aicc = zeros(size(data["Ξtrue"])..., runs)
        Ξs_slic = similar(Ξs_aicc)
        for run=1:runs
            # add noise
            η = NoisePct*std(umat)/100  
            un = umat + η.*randn(size(umat))
            usm = smooth_pde(un)
            # define lib 
            function SGLib(u, doms, ws, ps, s)
                #X = vec(u)
                pt,px,py = ps
                ws = wind*[1,1,1]
                f1 = vec(wInt_sc_2d_v2(ones(size(u)), doms... , ws , pt=pt,px=px,py=py, Δ=s))
                f2 = vec(dtwInt_sc_2d_v2(u, doms... ,  ws  , pt=pt,px=px,py=py, Δ=s))
                f3 = vec(dxwInt_sc_2d_v2(u, doms... ,  ws  , pt=pt,px=px,py=py, Δ=s))
                f4 = vec(dywInt_sc_2d_v2(u, doms... ,  ws  , pt=pt,px=px,py=py, Δ=s))
                f5 = vec(d2xwInt_sc_2d_v2(u, doms... ,  ws  , pt=pt,px=px,py=py, Δ=s))
                f6 = vec(d2ywInt_sc_2d_v2(u, doms... ,  ws  , pt=pt,px=px,py=py, Δ=s))
                f7 = vec(dxywInt_sc_2d_v2(u, doms... , ws  , pt=pt,px=px,py=py, Δ=s))
                f8 = vec(wInt_sc_2d_v2(u , doms... ,  ws  , pt=pt,px=px,py=py, Δ=s))
                f9 = vec(wInt_sc_2d_v2(u.^2, doms... ,  ws  , pt=pt,px=px,py=py, Δ=s))
                f10 = vec(wInt_sc_2d_v2(sin.(u), doms... ,  ws  , pt=pt,px=px,py=py, Δ=s))
                f11 = vec(wInt_sc_2d_v2(cos.(u), doms... ,  ws  , pt=pt,px=px,py=py, Δ=s))

                return [f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11]
            end
            # set polynomial order and determine SSIM-opt window 
            ps = (8,8)
            _,_,wind = FindW(usm[:,:,1], tmat, xmat, ps, ws=25:2:55)  

            # peform galerkin-projection 
            qt = vec(d2twInt_sc_2d_v2(usm, doms..., wind*[1,1,1], pt=8,px=8,py=8, Δ=s))
            qt = reshape(qt, (length(qt), 1))
            θ = iSGLib(usm, doms , wind , ps, s)

            Ξslic, _ = EnAdSR(θ, qt, "slic", tol=tol, c=c, num_batches=num_batches)
            Ξaicc, _ = EnAdSR(θ, qt, "aicc", tol=tol, c=c, num_batches=num_batches)

            Ξs_aicc[:,:,run] = Ξaicc
            Ξs_slic[:,:,run] = Ξslic
        end
        return data["Ξtrue"], Ξs_aicc, Ξs_slic
    end
end

# Disclaimer: this code is certainly not optimized! The galerkin projection is the rate-limiting step. Would be smarter to parallelize the operations/use convolutions   
# If using sys=5, set s=5, or it will run quite long 

sys = 1 
NoisePct = 20
runs = 5 # number of times we randomly instantiate the noise at fixed noise %
Ξtrue, Ξaiccs, Ξslics = DiscoverPDEModel(sys, NoisePct, runs, s=1)

# visualize the mean of the models across the different noise instantiations  
paic = VisResults(Ξtrue, mean(Ξaiccs, dims=3)[:,:], "AICc")
pslic = VisResults(Ξtrue, mean(Ξslics, dims=3)[:,:], "SLIC")

display(paic)
display(pslic)