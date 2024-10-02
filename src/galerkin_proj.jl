using Trapz , ForwardDiff

# Uses SSIM to determine a window (at fixed p) that optimally preserves features of the underlying signal
# INPUTS: 
# X = data 
# ts, xs = temporal, spatial grids  
# p, ps = polynomial order for weighting function (see below)
# ws = window sizes to sweep  

# OUTPUTS: 
# ws => windows 
# ssims => SSIM scores
# wbest => window with best (largest) SSIM score

function FindW(X::AbstractMatrix, ts::AbstractVector;
        ws = 21:2:121,
        p = 10
    )
    
    n = size(X,1) # number of vars 
    ssims = zeros(length(ws))
    for (i, wind) in enumerate(ws)
        ssim_w = 0
        w_hf = Int(ceil(wind/2))
        Xtr = X[:, w_hf:end-w_hf]
        for j=1:n
            Xint = wInt_sc(X[j,:], ts, wind, p=p) 
            ssim_w += ssim(Xtr[j,:], Xint)
        end

        ssims[i] = ssim_w/n
        
    end

    ssimmax , imax = findmax(ssims) 
    wmax = ws[imax]
    wbest = wmax
    for i=imax+1:length(ssims)
        if abs(ssims[i] - ssimmax) < 0.05 # choose the best ssim that's within X% (5% here) of the max
            wbest = ws[i]
        end
    end

    return ws , ssims , wbest
end

# for higher-dim inputs
function FindW(X, ts, xs, ps;
        ws = 21:2:55,
        s=1
    )
    
    n = size(X,3) # number of vars 
    ssims = zeros(length(ws))
    
    for (i,wind) in enumerate(ws)
        ssim_w = 0
        w_hf = Int(ceil(wind/2))
        Xtr = X[w_hf:end-w_hf, w_hf:end-w_hf, :]
        for j=1:n
            #Xint = wInt_sc(X[:,:,j], ts, xs, wind*[1,1], pt=pt, px=px)
            Xint  = IntData(X[:,:,j], (ts, xs), (0,0) ,wind, ps, s)
            ssim_w += ssim(Xtr[:,:,j], Xint)
        end

        ssims[i] = ssim_w/n
        
    end

    ssimmax , imax = findmax(ssims) 
    wmax = ws[imax]
    wbest = wmax
    for i=imax+1:length(ssims)
        if abs(ssims[i] - ssimmax) < 0.05 # choose the best ssim that's within 10% of the max
            wbest = ws[i]
        end
    end

    return ws , ssims , wbest
end

# SSIM scoring
function ssim(x,y)
    L = length(x)
    μx, μy = mean(x), mean(y) # compute means 
    σx, σy = std(x), std(y) # compute stds
    σxy = cov(x[:],y[:]) # compute covariance
    c1 = 0.01*L 
    c2 = c1/2
    return ((2*μx*μy + c1)/(μx^2 + μy^2 + c1))*((2*σx*σy + c1)/(σx^2 + σy^2 + c1))*((σxy + c2)/(σx*σy + c2))
end

# define the weight functions for the integration 
w(t,p,a,b) = @. (1/(p)^(2*p))*((2p/(b-a))^(2p))*((t-a)*(b-t))^p
dw(t,p,a,b) = @. ForwardDiff.derivative(x -> w(x,p,a,b), t)
d2w(t,p,a,b) = @. ForwardDiff.derivative(x -> dw(x,p,a,b), t)
d3w(t,p,a,b) = @. ForwardDiff.derivative(x -> d2w(x,p,a,b), t)
d4w(t,p,a,b) = @. ForwardDiff.derivative(x -> d3w(x,p,a,b), t)

# Code for trapezoidal integration for window of a given size 
# Disclaimer: definitely not optimized for performance! Lots of repeated code. 

# =================== For ODEs ===================
# ys = data 
# ts = temporal grids 
# ws = vector of window sizes for each grid 
# p = polynomial order for weighting functions 
# Δ = 'downsampling'/'subsampling' factor, which determines how many datapoints are skipped during window sliding 
function wInt_sc(ys::AbstractVector, ts,  wind::Int; p=10, Δ=1)
    # ys = smoothed data
    # ts = timepoints
    # w = window size 
    T = 1:Δ:(length(ts)-wind)
    q = zeros(length(T))
    for (i,t) in enumerate(T)
        # time window 
        ts_i = ts[t:t+wind]
        # scale the domain to [-1, 1]
        sc_ts_i = ts_i*(2/(ts_i[end]-ts_i[1])) .- (ts_i[end]+ts_i[1])/(ts_i[end]-ts_i[1])
        # getting values on that window 
        ys_i = ys[t:t+wind]
        # determine the function        
        f_i = ys_i .* w(sc_ts_i, p, -1, 1) / (2/(ts_i[end]-ts_i[1]))
        q[i] = trapz(sc_ts_i, f_i)
    end

    return q
end

function dwInt_sc(ys::AbstractVector, ts,  wind::Int; p=10, Δ=1)
    # ys = smoothed data
    # ts = timepoints
    # w = window size 
    #T = 1:Δ:(length(ts)-wind) # starting indices for integration window 
    T = range(1, length(ts)-wind, step=Δ)
    q = zeros(length(T))
    for (i, t) in enumerate(T)
        # time window 
        ts_i = ts[t:t+wind]
        # scale the domain to [-1, 1]
        sc_ts_i = ts_i*(2/(ts_i[end]-ts_i[1])) .- (ts_i[end]+ts_i[1])/(ts_i[end]-ts_i[1])
        # getting values on that window 
        ys_i = ys[t:t+wind]
        # determine the function        
        f_i = ys_i .* -dw(sc_ts_i, p, -1, 1) #/ (2/(ts_i[end]-ts_i[1]))
        q[i] = trapz(sc_ts_i, f_i)
    end

    return q
end

function dwInt_sc(ys::AbstractMatrix, ts, wind::Int; p=10, Δ=1)
    # ys = smoothed data
    # ts = timepoints
    # w = window size 
    qt = dwInt_sc(ys[1,:], ts, wind, p=p, Δ=Δ)
    for i=2:size(ys,1)
        qt = hcat(qt, dwInt_sc(ys[i,:], ts, wind, p=p, Δ=Δ))
    end
    
    return qt'
end

function d2wInt_sc(ys::AbstractVector, ts,  wind::Int; p=10, Δ=1)
    # ys = smoothed data
    # ts = timepoints
    # w = window size 
    #T = 1:Δ:(length(ts)-wind) # starting indices for integration window 
    T = range(1, length(ts)-wind, step=Δ)
    q = zeros(length(T))
    for (i, t) in enumerate(T)
        # time window 
        ts_i = ts[t:t+wind]
        # scale the domain to [-1, 1]
        sc_ts_i = ts_i*(2/(ts_i[end]-ts_i[1])) .- (ts_i[end]+ts_i[1])/(ts_i[end]-ts_i[1])
        # getting values on that window 
        ys_i = ys[t:t+wind]
        # determine the function        
        f_i = ys_i .* d2w(sc_ts_i, p, -1, 1) * ((ts_i[end]-ts_i[1])/2)^-1
        q[i] = trapz(sc_ts_i, f_i)
    end

    return q
end

function d2wInt_sc(ys::AbstractMatrix, ts, wind::Int; p=10, Δ=1)
    # ys = smoothed data
    # ts = timepoints
    # w = window size 
    qt = d2wInt_sc(ys[1,:], ts, wind, p=p, Δ=Δ)
    for i=2:size(ys,1)
        qt = hcat(qt, d2wInt_sc(ys[i,:], ts, wind, p=p, Δ=Δ))
    end
    
    return qt'
end

# =================== For PDEs ===================
# 1 space, 1 time dim 
# u = data 
# ts , xs, .. = temporal, spatial grids 
# ws = vector of window sizes for each grid 
# pt, px, ... = polynomial order for weighting functions for each dimension
# Δ, s = 'downsampling'/'subsampling' factor, which determines how many datapoints are skipped during window sliding 

function wInt_sc(u, ts, xs, ws::AbstractVector; pt=8, px=8)
    # ys = smoothed data
    # ts = timepoints
    # w = window size 
    wt = ws[1]
    wx = ws[2]
    q = zeros((length(xs)-wx, length(ts)-wt))

    for i=1:(length(xs)-wx)
        for j=1:(length(ts)-wt)
            # create x window & weights
            xs_i = xs[i:i+wx]
            sc_xs_i = xs_i*(2/(xs_i[end]-xs_i[1])) .- (xs_i[end]+xs_i[1])/(xs_i[end]-xs_i[1])

            wxs = w(sc_xs_i, px, -1, 1)
            # create t window & weighrs
            ts_j = ts[j:j+wt]
            sc_ts_j = ts_j*(2/(ts_j[end]-ts_j[1])) .- (ts_j[end]+ts_j[1])/(ts_j[end]-ts_j[1])

            wts = w(sc_ts_j, pt, -1, 1)
            # create grids 
            wxgrid = repeat(wxs, 1, length(sc_ts_j))
            wtgrid = repeat(wts, 1, length(sc_xs_i))'
            # create vals 
            u_i = u[i:i+wx, j:j+wt]
            f_i = u_i .* wxgrid .* wtgrid * (ts_j[end]-ts_j[1])/2 * (xs_i[end]-xs_i[1])/2 
            
            q[i,j] = trapz((sc_ts_j, sc_xs_i), f_i)
        end
    end
    return q
end

function dtwInt_sc(u, ts, xs, ws::AbstractVector; pt=8, px=8)
    # ys = smoothed data
    # ts = timepoints
    # w = window size 
    wt = ws[1]
    wx = ws[2]
    q = zeros((length(xs)-wx, length(ts)-wt))

    for i=1:(length(xs)-wx)
        for j=1:(length(ts)-wt)
            # create x window & weights
            xs_i = xs[i:i+wx]
            sc_xs_i = xs_i*(2/(xs_i[end]-xs_i[1])) .- (xs_i[end]+xs_i[1])/(xs_i[end]-xs_i[1])

            wxs = w(sc_xs_i, px, -1, 1)
            # create t window & weighrs
            ts_j = ts[j:j+wt]
            sc_ts_j = ts_j*(2/(ts_j[end]-ts_j[1])) .- (ts_j[end]+ts_j[1])/(ts_j[end]-ts_j[1])

            wts = -dw(sc_ts_j, pt, -1, 1)
            # create grids 
            wxgrid = repeat(wxs, 1, length(sc_ts_j))
            wtgrid = repeat(wts, 1, length(sc_xs_i))'
            # create vals 
            u_i = u[i:i+wx, j:j+wt]
            f_i = u_i .* wxgrid .* wtgrid * (xs_i[end]-xs_i[1])/2
            
            q[i,j] = trapz((sc_ts_j, sc_xs_i), f_i)
        end
    end
    return q
end

function dxwInt_sc(u, ts, xs, ws::AbstractVector; pt=8, px=8)
    # ys = smoothed data
    # ts = timepoints
    # w = window size 
    wt = ws[1]
    wx = ws[2]
    q = zeros((length(xs)-wx, length(ts)-wt))

    for i=1:(length(xs)-wx)
        for j=1:(length(ts)-wt)
            # create x window & weights
            xs_i = xs[i:i+wx]
            sc_xs_i = xs_i*(2/(xs_i[end]-xs_i[1])) .- (xs_i[end]+xs_i[1])/(xs_i[end]-xs_i[1])

            wxs = -dw(sc_xs_i, px, -1, 1)
            # create t window & weighrs
            ts_j = ts[j:j+wt]
            sc_ts_j = ts_j*(2/(ts_j[end]-ts_j[1])) .- (ts_j[end]+ts_j[1])/(ts_j[end]-ts_j[1])

            wts = w(sc_ts_j, pt, -1, 1)
            # create grids 
            wxgrid = repeat(wxs, 1, length(sc_ts_j))
            wtgrid = repeat(wts, 1, length(sc_xs_i))'
            # create vals 
            u_i = u[i:i+wx, j:j+wt]
            f_i = u_i .* wxgrid .* wtgrid * (ts_j[end]-ts_j[1])/2
            
            q[i,j] = trapz((sc_ts_j, sc_xs_i), f_i)
        end
    end
    return q
end

function d2xwInt_sc(u, ts, xs, ws::AbstractVector; pt=8, px=8)
    # ys = smoothed data
    # ts = timepoints
    # w = window size 
    wt = ws[1]
    wx = ws[2]
    q = zeros((length(xs)-wx, length(ts)-wt))

    for i=1:(length(xs)-wx)
        for j=1:(length(ts)-wt)
            # create x window & weights
            xs_i = xs[i:i+wx]
            sc_xs_i = xs_i*(2/(xs_i[end]-xs_i[1])) .- (xs_i[end]+xs_i[1])/(xs_i[end]-xs_i[1])

            wxs = d2w(sc_xs_i, px, -1, 1)
            # create t window & weighrs
            ts_j = ts[j:j+wt]
            sc_ts_j = ts_j*(2/(ts_j[end]-ts_j[1])) .- (ts_j[end]+ts_j[1])/(ts_j[end]-ts_j[1])

            wts = w(sc_ts_j, pt, -1, 1)
            # create grids 
            wxgrid = repeat(wxs, 1, length(sc_ts_j))
            wtgrid = repeat(wts, 1, length(sc_xs_i))'
            # create vals 
            u_i = u[i:i+wx, j:j+wt]
            f_i = u_i .* wxgrid .* wtgrid * (ts_j[end]-ts_j[1])/2 * ((xs_i[end]-xs_i[1])/2 )^-1
            
            q[i,j] = trapz((sc_ts_j, sc_xs_i), f_i)
        end
    end
    return q
end

function d3xwInt_sc(u, ts, xs, ws::AbstractVector; pt=8, px=8)
    # ys = smoothed data
    # ts = timepoints
    # w = window size 
    wt = ws[1]
    wx = ws[2]
    q = zeros((length(xs)-wx, length(ts)-wt))

    for i=1:(length(xs)-wx)
        for j=1:(length(ts)-wt)
            # create x window & weights
            xs_i = xs[i:i+wx]
            sc_xs_i = xs_i*(2/(xs_i[end]-xs_i[1])) .- (xs_i[end]+xs_i[1])/(xs_i[end]-xs_i[1])

            wxs = -d3w(sc_xs_i, px, -1, 1)
            # create t window & weighrs
            ts_j = ts[j:j+wt]
            sc_ts_j = ts_j*(2/(ts_j[end]-ts_j[1])) .- (ts_j[end]+ts_j[1])/(ts_j[end]-ts_j[1])

            wts = w(sc_ts_j, pt, -1, 1)
            # create grids 
            wxgrid = repeat(wxs, 1, length(sc_ts_j))
            wtgrid = repeat(wts, 1, length(sc_xs_i))'
            # create vals 
            u_i = u[i:i+wx, j:j+wt]
            f_i = u_i .* wxgrid .* wtgrid * (ts_j[end]-ts_j[1])/2 * ((xs_i[end]-xs_i[1])/2 )^-2
            
            q[i,j] = trapz((sc_ts_j, sc_xs_i), f_i)
        end
    end
    return q
end

function d4xwInt_sc(u, ts, xs, ws::AbstractVector; pt=8, px=8)
    # ys = smoothed data
    # ts = timepoints
    # w = window size 
    wt = ws[1]
    wx = ws[2]
    q = zeros((length(xs)-wx, length(ts)-wt))

    for i=1:(length(xs)-wx)
        for j=1:(length(ts)-wt)
            # create x window & weights
            xs_i = xs[i:i+wx]
            sc_xs_i = xs_i*(2/(xs_i[end]-xs_i[1])) .- (xs_i[end]+xs_i[1])/(xs_i[end]-xs_i[1])
            wxs = d4w(sc_xs_i, px, -1, 1)
            # create t window & weighrs
            ts_j = ts[j:j+wt]
            sc_ts_j = ts_j*(2/(ts_j[end]-ts_j[1])) .- (ts_j[end]+ts_j[1])/(ts_j[end]-ts_j[1])
            wts = w(sc_ts_j, pt, -1, 1)
            # create grids 
            wxgrid = repeat(wxs, 1, length(sc_ts_j))
            wtgrid = repeat(wts, 1, length(sc_xs_i))'
            # create vals 
            u_i = u[i:i+wx, j:j+wt]
            f_i = u_i .* wxgrid .* wtgrid * (ts_j[end]-ts_j[1])/2 * ((xs_i[end]-xs_i[1])/2 )^-3
            
            q[i,j] = trapz((sc_ts_j, sc_xs_i), f_i)
        end
    end
    return q
end

# Smarter integrated way to integrate the above, but buggy for higher spatial dims 
function IntData(
    u::Array{Float64, N}, # input data, with domains arranged by: time, x1, x2, ....
    doms::NTuple{N, Vector{Float64}} , # domains in order  
    codes::NTuple{N, Int64} , # derivate codes for weights and scaling factors: 0 = no deriv, 1 = 1st deriv, etc..
    wind::Int64, # window size for integration
    ps::NTuple{N, Int64} , # polynomial order for weight functions
    s::Int64 # skipping factor for subsampling 
    ) where N

    d = ndims(u) # dimensionality of u 
    # allocate for returned matrix
    q = zeros(Tuple(length(doms[i]) - wind for i=1:d))
    #q = zeros((length(xs)-wind, length(ts)-w))

    # create arrays and grids
    dom_sc = range(-1,1, wind+1) # scaled domain array 
    wfs, scfs = CodesToWeightAndScaleFuncs(codes) # get functions for integral weights and scale factors
    wgrid = repeat(wfs[1](dom_sc, ps[1], -1,1), (1,ntuple(x -> wind+1, d-1)...)...)
    #wgrid = ones((length(dom_sc), d))
    for dim=2:d
        # create xi grid, which is repeated along columns 
        xi_grid = repeat(wfs[dim](dom_sc, ps[dim], -1,1), (1,ntuple(x -> wind+1, d-1)...)...)
        # need to re-align to proper axis by permuting the array
        xi_grid = permutedims(xi_grid, Tuple(circshift(1:d, d-1)))
        wgrid .*= xi_grid # update the grid 
        #display(wgrid)
    end
    # create matrix of coordinates to start integrating from
    CM = collect(Iterators.product((1:s:length(doms[i])-wind for i=1:d)...))
    @simd for cd in CM
        f_i = u[(cd[i]:cd[i]+wind for i=1:d)...] .* wgrid * prod(scfs[i](doms[i][cd[i]:cd[i]+wind]) for i=1:d)
        #display(f_i)
        q[cd...] = trapz(ntuple(x -> dom_sc, d), f_i)
    end
    #return q[(1:s:length(doms[i])-wind for i=1:d)...], wgrid
    return q[(1:s:length(doms[i])-wind for i=1:d)...]#, wgrid

end

function CodesToWeightAndScaleFuncs(codes::NTuple{N, Int64}) where N
    wfs = Tuple(CodeToWeightFunc(i) for i in codes)
    scfs = Tuple(CodeToScaleFunc(i) for i in codes)
    return wfs, scfs
end

function CodeToWeightFunc(code::Int64)
    if code == 0
        return w
    elseif code == 1
        return (x -> -x) ∘ dw # this just composes a signflip with the derivative function 
    elseif code == 2
        return d2w 
    elseif code == 3
        return (x -> -x) ∘ d3w
    elseif code == 4
        return d4w
    else
        error("Must return an integer between 0 and 4!")
    end
end

function CodeToScaleFunc(code::Int64)
    if 0 <= code <= 4
        return x -> ((x[end]-x[1])/2)^(1-code)
    else
        error("Must return an integer between 0 and 4!")
    end
end

# For 2 spatial dims, 1 time dim 
function wInt_sc(u, ts, xs, ys, ws::AbstractVector; pt=8, px=8, py=8, Δ=1)
    # ys = smoothed data
    # ts = timepoints
    # w = window size 
    wt = ws[1]
    wx = ws[2]
    wy = ws[3]
    wind = wx
    tcs = 1:Δ:(length(ts)-wt)
    xcs = 1:Δ:(length(xs)-wx)
    ycs = 1:Δ:(length(ys)-wy)
    q = zeros((length(ys)-wy, length(xs)-wx, length(ts)-wt))
    #q = zeros((length(ycs), length(xcs), length(tcs)))
    dom_sc = range(-1,1, wind+1)

    wys = w(dom_sc, py, -1, 1)
    wygrid = repeat(wys, 1, wind+1, wind+1)

    wxs = w(dom_sc, px, -1, 1)
    wxgrid = repeat(wxs, 1, wind+1, wind+1)
    wxgrid = permutedims(wxgrid, (3,1,2))

    wts = w(dom_sc, pt, -1, 1)
    wtgrid = repeat(wts, 1, wind+1, wind+1)
    wtgrid = permutedims(wtgrid, (2,3,1))

    wgrid = wygrid .* wxgrid .* wtgrid

    for i in ycs
        for j in xcs
            for k in tcs

                u_i = u[i:i+wy, j:j+wx, k:k+wt]
                f_i = @. u_i * wgrid * (ts[k+wt]-ts[k])/2 * (xs[j+wx]-xs[j])/2 * (ys[i+wy]-ys[i])/2 
                
                q[i,j,k] = trapz((dom_sc, dom_sc, dom_sc), f_i)
            end
        end
    end
    return q[ycs, xcs, tcs]
end

function dtwInt_sc(u, ts, xs, ys, ws::AbstractVector; pt=8, px=8, py=8, Δ=1)
    # ys = smoothed data
    # ts = timepoints
    # w = window size 
    wt = ws[1]
    wx = ws[2]
    wy = ws[3]
    wind = wx

    tcs = 1:Δ:(length(ts)-wt)
    xcs = 1:Δ:(length(xs)-wx)
    ycs = 1:Δ:(length(ys)-wy)
    q = zeros((length(ys)-wy, length(xs)-wx, length(ts)-wt))
    #q = zeros((length(ycs), length(xcs), length(tcs)))
    dom_sc = range(-1,1, wx+1)

    wys = w(dom_sc, py, -1, 1)
    wygrid = repeat(wys, 1, wy+1, wy+1)

    wxs = w(dom_sc, px, -1, 1)
    wxgrid = repeat(wxs, 1, wx+1, wx+1)
    wxgrid = permutedims(wxgrid, (3,1,2))

    wts = -dw(dom_sc, pt, -1, 1)
    wtgrid = repeat(wts, 1, wind+1, wind+1)
    wtgrid = permutedims(wtgrid, (2,3,1))

    wgrid = wygrid .* wxgrid .* wtgrid

    for i in ycs
        for j in xcs
            for k in tcs

                u_i = u[i:i+wy, j:j+wx, k:k+wt]
                f_i = @. u_i * wgrid * (xs[j+wx]-xs[j])/2 * (ys[i+wy]-ys[i])/2 
                
                q[i,j,k] = trapz((dom_sc, dom_sc, dom_sc), f_i)
            end
        end
    end
    return q[ycs, xcs, tcs]
end

function dxwInt_sc(u, ts, xs, ys, ws::AbstractVector; pt=8, px=8, py=8, Δ=1)
    # ys = smoothed data
    # ts = timepoints
    # w = window size 
    wt = ws[1]
    wx = ws[2]
    wy = ws[3]
    wind=wx
    tcs = 1:Δ:(length(ts)-wt)
    xcs = 1:Δ:(length(xs)-wx)
    ycs = 1:Δ:(length(ys)-wy)
    q = zeros((length(ys)-wy, length(xs)-wx, length(ts)-wt))
    #q = zeros((length(ycs), length(xcs), length(tcs)))
    dom_sc = range(-1,1, wind+1)

    wys = w(dom_sc, py, -1, 1)
    wygrid = repeat(wys, 1, wind+1, wind+1)

    wxs = -dw(dom_sc, px, -1, 1)
    wxgrid = repeat(wxs, 1, wind+1, wind+1)
    wxgrid = permutedims(wxgrid, (3,1,2))

    wts = w(dom_sc, pt, -1, 1)
    wtgrid = repeat(wts, 1, wind+1, wind+1)
    wtgrid = permutedims(wtgrid, (2,3,1))

    wgrid = wygrid .* wxgrid .* wtgrid

    for i in ycs
        for j in xcs
            for k in tcs

                u_i = u[i:i+wy, j:j+wx, k:k+wt]
                f_i = @. u_i * wgrid * (ts[k+wt]-ts[k])/2 * (ys[i+wy]-ys[i])/2 
                
                q[i,j,k] = trapz((dom_sc, dom_sc, dom_sc), f_i)
            end
        end
    end
    return q[ycs, xcs, tcs]
end

function dywInt_sc(u, ts, xs, ys, ws::AbstractVector; pt=8, px=8, py=8, Δ=1)
    # ys = smoothed data
    # ts = timepoints
    # w = window size 
    wt = ws[1]
    wx = ws[2]
    wy = ws[3]
    wind=wx
    tcs = 1:Δ:(length(ts)-wt)
    xcs = 1:Δ:(length(xs)-wx)
    ycs = 1:Δ:(length(ys)-wy)
    q = zeros((length(ys)-wy, length(xs)-wx, length(ts)-wt))
    #q = zeros((length(ycs), length(xcs), length(tcs)))
    dom_sc = range(-1,1, wind+1)

    wys = -dw(dom_sc, py, -1, 1)
    wygrid = repeat(wys, 1, wind+1, wind+1)

    wxs = w(dom_sc, px, -1, 1)
    wxgrid = repeat(wxs, 1, wind+1, wind+1)
    wxgrid = permutedims(wxgrid, (3,1,2))

    wts = w(dom_sc, pt, -1, 1)
    wtgrid = repeat(wts, 1, wind+1, wind+1)
    wtgrid = permutedims(wtgrid, (2,3,1))

    wgrid = wygrid .* wxgrid .* wtgrid

    for i in ycs
        for j in xcs
            for k in tcs

                u_i = u[i:i+wy, j:j+wx, k:k+wt]
                f_i = @. u_i * wgrid * (ts[k+wt]-ts[k])/2 * (xs[j+wx]-xs[j])/2 
                
                q[i,j,k] = trapz((dom_sc, dom_sc, dom_sc), f_i)
            end
        end
    end
    return q[ycs, xcs, tcs]
end

function dxywInt_sc(u, ts, xs, ys, ws::AbstractVector; pt=8, px=8, py=8, Δ=1)
    # ys = smoothed data
    # ts = timepoints
    # w = window size 
    wt = ws[1]
    wx = ws[2]
    wy = ws[3]
    wind=wx
    tcs = 1:Δ:(length(ts)-wt)
    xcs = 1:Δ:(length(xs)-wx)
    ycs = 1:Δ:(length(ys)-wy)
    q = zeros((length(ys)-wy, length(xs)-wx, length(ts)-wt))
    #q = zeros((length(ycs), length(xcs), length(tcs)))
    dom_sc = range(-1,1, wind+1)

    wys = -dw(dom_sc, py, -1, 1)
    wygrid = repeat(wys, 1, wind+1, wind+1)

    wxs = -dw(dom_sc, px, -1, 1)
    wxgrid = repeat(wxs, 1, wind+1, wind+1)
    wxgrid = permutedims(wxgrid, (3,1,2))

    wts = w(dom_sc, pt, -1, 1)
    wtgrid = repeat(wts, 1, wind+1, wind+1)
    wtgrid = permutedims(wtgrid, (2,3,1))

    wgrid = wygrid .* wxgrid .* wtgrid

    for i in ycs
        for j in xcs
            for k in tcs 

                u_i = u[i:i+wy, j:j+wx, k:k+wt]
                f_i = @. u_i * wgrid * (ts[k+wt]-ts[k])/2 
                
                q[i,j,k] = trapz((dom_sc, dom_sc, dom_sc), f_i)
            end
        end
    end
    return q[ycs, xcs, tcs]
end

function d2twInt_sc(u, ts, xs, ys, ws::AbstractVector; pt=8, px=8, py=8, Δ=1)
    # ys = smoothed data
    # ts = timepoints
    # w = window size 
    wt = ws[1]
    wx = ws[2]
    wy = ws[3]
    wind=wx
    tcs = 1:Δ:(length(ts)-wt)
    xcs = 1:Δ:(length(xs)-wx)
    ycs = 1:Δ:(length(ys)-wy)
    q = zeros((length(ys)-wy, length(xs)-wx, length(ts)-wt))
    #q = zeros((length(ycs), length(xcs), length(tcs)))
    dom_sc = range(-1,1, wind+1)

    wys = w(dom_sc, py, -1, 1)
    wygrid = repeat(wys, 1, wind+1, wind+1)

    wxs = w(dom_sc, px, -1, 1)
    wxgrid = repeat(wxs, 1, wind+1, wind+1)
    wxgrid = permutedims(wxgrid, (3,1,2))

    wts = d2w(dom_sc, pt, -1, 1)
    wtgrid = repeat(wts, 1, wind+1, wind+1)
    wtgrid = permutedims(wtgrid, (2,3,1))

    wgrid = wygrid .* wxgrid .* wtgrid

    for i in ycs
        for j in xcs
            for k in tcs
                
                u_i = u[i:i+wy, j:j+wx, k:k+wt]
                f_i = @. u_i * wgrid * ((ts[k+wt]-ts[k])/2)^-1 * (xs[j+wx]-xs[j])/2 * (ys[i+wy]-ys[i])/2 
                
                q[i,j,k] = trapz((dom_sc, dom_sc, dom_sc), f_i)
            end
        end
    end
    return q[ycs, xcs, tcs]
end

function d2xwInt_sc(u, ts, xs, ys, ws::AbstractVector; pt=8, px=8, py=8, Δ=1)
    # ys = smoothed data
    # ts = timepoints
    # w = window size 
    wt = ws[1]
    wx = ws[2]
    wy = ws[3]
    wind=wx
    tcs = 1:Δ:(length(ts)-wt)
    xcs = 1:Δ:(length(xs)-wx)
    ycs = 1:Δ:(length(ys)-wy)
    q = zeros((length(ys)-wy, length(xs)-wx, length(ts)-wt))
    #q = zeros((length(ycs), length(xcs), length(tcs)))
    dom_sc = range(-1,1, wind+1)

    wys = w(dom_sc, py, -1, 1)
    wygrid = repeat(wys, 1, wind+1, wind+1)

    wxs = d2w(dom_sc, px, -1, 1)
    wxgrid = repeat(wxs, 1, wind+1, wind+1)
    wxgrid = permutedims(wxgrid, (3,1,2))

    wts = w(dom_sc, pt, -1, 1)
    wtgrid = repeat(wts, 1, wind+1, wind+1)
    wtgrid = permutedims(wtgrid, (2,3,1))

    wgrid = wygrid .* wxgrid .* wtgrid

    for i in ycs
        for j in xcs
            for k in tcs

                u_i = u[i:i+wy, j:j+wx, k:k+wt]
                f_i = @. u_i * wgrid * (ts[k+wt]-ts[k])/2 * ((xs[j+wx]-xs[j])/2)^-1 * (ys[i+wy]-ys[i])/2 
                
                q[i,j,k] = trapz((dom_sc, dom_sc, dom_sc), f_i)
            end
        end
    end
    return q[ycs, xcs, tcs]
end

function d2ywInt_sc(u, ts, xs, ys, ws::AbstractVector; pt=8, px=8, py=8, Δ=1)
    # ys = smoothed data
    # ts = timepoints
    # w = window size 
    wt = ws[1]
    wx = ws[2]
    wy = ws[3]
    wind=wx
    tcs = 1:Δ:(length(ts)-wt)
    xcs = 1:Δ:(length(xs)-wx)
    ycs = 1:Δ:(length(ys)-wy)
    q = zeros((length(ys)-wy, length(xs)-wx, length(ts)-wt))
    #q = zeros((length(ycs), length(xcs), length(tcs)))
    dom_sc = range(-1,1, wind+1)

    wys = d2w(dom_sc, py, -1, 1)
    wygrid = repeat(wys, 1, wind+1, wind+1)

    wxs = w(dom_sc, px, -1, 1)
    wxgrid = repeat(wxs, 1, wind+1, wind+1)
    wxgrid = permutedims(wxgrid, (3,1,2))

    wts = w(dom_sc, pt, -1, 1)
    wtgrid = repeat(wts, 1, wind+1, wind+1)
    wtgrid = permutedims(wtgrid, (2,3,1))

    wgrid = wygrid .* wxgrid .* wtgrid

    for i in ycs
        for j in xcs
            for k in tcs

                u_i = u[i:i+wy, j:j+wx, k:k+wt]
                f_i = @. u_i * wgrid * (ts[k+wt]-ts[k])/2 * (xs[j+wx]-xs[j])/2 * ((ys[i+wy]-ys[i])/2)^-1 
                
                q[i,j,k] = trapz((dom_sc, dom_sc, dom_sc), f_i)
            end
        end
    end
    return q[ycs, xcs, tcs]
end

