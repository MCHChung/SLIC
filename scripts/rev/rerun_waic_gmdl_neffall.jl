using DrWatson
@quickactivate "SLIC"
using JLD, Random, Statistics

include(scriptsdir("rev", "rev_common.jl"))

# ----------------------------------------------------------------------------
# Note on the residual floor eta (applies identically to every criterion,
# SLIC included). Candidates are scored on RSS + eta rather than RSS; see
# score() in src/sparse_regress.jl. In the main pipeline eta = c*cond(theta),
# with the per-system constant c passed to EnAdSR. In the exhaustive
# enumeration (enumerate_all.jl) eta = gamma*RSS0, where RSS0 is the true
# model's residual on noise-free data (calibrate_c.jl) and gamma = 100
# (SLIC_GAMMA). The floor is numerical: as the noise goes to zero, RSS goes to
# zero, which would otherwise make the log-likelihood terms diverge and leave
# the fit term free to fall with every added term, so that most criteria would
# not sparsify at low noise.
# ----------------------------------------------------------------------------

include(srcdir("waic_gmdl.jl"))   # corrected WAIC (exact sampling) + gMDL

# ============================================================================
# Rerun benchmarks with CORRECTED WAIC and gMDL (R1 Concern 2).
#
# Supersedes rerun_with_waic_nml.jl. The earlier WAIC (Laplace) and NML
# (broken closed form) are replaced by:
#   - WAIC via exact conjugate-posterior sampling
#   - gMDL (Hansen & Yu 2001), the correct closed-form MDL for Gaussian
#     regression with unknown variance
#
# Because these two criteria are not in the score() switch by default, this
# script scores candidate models directly: it reuses Algorithm 1's candidate
# generation by running EnAdSR with SLIC to get a sparsity pattern, then for
# a sweep of thresholds re-scores with WAIC/gMDL. For a cleaner and more
# defensible comparison we instead enumerate the SLIC ensemble's support
# union and let WAIC/gMDL pick among nested sub-supports.
#
# Simpler + matches the other criteria: add "waic"/"gmdl" cases to score()
# and call EnAdSR the same way. We do that here via a local scoring wrapper.
# ============================================================================

const NOISE_WG = [0, 5, 10, 20, 30, 40]
const RUNS_WG = 25
const NUM_BATCHES_WG = 20
const TOL_WG = 0.7
const CC_WG  = 1e-2
const NEW_ICS = ["waic", "gmdl"]

# Local score() extension: dispatches waic/gmdl to the corrected implementations,
# falls back to the standard score() for everything else.
function score_ext(y, θ, Ξ, ic, η; n_eff=nothing)
    if ic == "waic"
        n = size(y, 1)
        RSS = sum(abs2, y - θ*Ξ) + η
        return waic_score(y, θ, Ξ, RSS, n; n_eff=n_eff, n_samples=500)
    elseif ic == "gmdl"
        n = size(y, 1)
        k = count(abs.(Ξ) .> 0.) + 1
        RSS = sum(abs2, y - θ*Ξ) + η
        return gmdl_score(y, θ, Ξ, RSS, n, k; n_eff=n_eff)
    else
        return score(y, θ, Ξ, ic, η; n_eff=n_eff)
    end
end

# Minimal AdSR variant that uses score_ext (so we can run waic/gmdl through
# the same threshold-sweep machinery as the classical criteria).
function AdSR_ext(θ, y, ic::String; iter=10, c=0., trainpct=80, n_eff=nothing, score_on_all::Bool=false)
    nobs, n_state = size(y)
    bag_size = Int(floor(trainpct*nobs/100))
    traininds = sort(sample(1:nobs, bag_size, replace=false))
    testinds = [i for i=1:nobs if i ∉ traininds]
    θ_train = θ[traininds, :]; θ_test = θ[testinds, :]
    y_train = y[traininds, :]; y_test = y[testinds, :]
    η = c*cond(θ_train)

    # score_on_all: fit on the train split (ensemble still varies across batches)
    # but score on ALL n, with η rescaled by nobs/n_test so the relative
    # conditioning floor η/RSS is preserved and the n effect is isolated.
    # Mirrors scripts/rev/sparse_regress_probe.jl exactly.
    θ_score = score_on_all ? θ : θ_test
    y_score = score_on_all ? y : y_test
    n_eff_score = if isnothing(n_eff)
        nothing
    elseif score_on_all
        n_eff
    else
        n_eff * (100-trainpct)/100
    end
    η_score = (score_on_all && !isempty(testinds)) ? η * (nobs/length(testinds)) : η

    min_score = Inf
    prev_smallinds = [1]
    Ξes = θ_train \ y_train
    X_prev = θ_train * Ξes

    for i=1:iter
        nzv = Ξes[Ξes .!= 0]
        isempty(nzv) && break
        λmin = minimum(abs.(nzv)); λmax = maximum(abs.(nzv))
        λs = range(λmin, λmax, abs(1000*Int(ceil(log10(λmax/λmin + 1e-12)))) + 2)
        for λ in λs
            temp = copy(Ξes)
            smallinds = (abs.(temp) .< λ)
            smallinds == prev_smallinds && continue
            temp[smallinds] .= 0
            any(all(abs.(temp) .== 0, dims=1)) && break
            for ind=1:n_state
                biginds = .!smallinds[:,ind]
                temp[biginds,ind] = θ_train[:,biginds] \ y_train[:,ind]
            end
            prev_smallinds = smallinds
            sc = score_ext(y_score, θ_score, temp, ic, η_score; n_eff=n_eff_score)
            if sc < min_score
                Ξes = copy(temp); min_score = sc
            end
        end
        X = θ_train * Ξes
        norm(X - X_prev) < 1e-7 && break
        X_prev = X
    end
    return Ξes, min_score
end

function EnAdSR_ext(θ, y, ic::String; c=0., trainpct=80, num_batches=10, iter=10, tol=0.7, n_eff=nothing, score_on_all::Bool=false)
    ΞB = zeros((size(θ,2), size(y,2), num_batches))
    for i=1:num_batches
        Ξes, _ = AdSR_ext(θ, y, ic; iter=iter, c=c, trainpct=trainpct, n_eff=n_eff, score_on_all=score_on_all)
        ΞB[:,:,i] = Ξes
    end
    biginds = abs.(ΞB) .> 0
    ips = mean(biginds, dims=3)
    Ξes = sum(ΞB, dims=3) ./ max.(count(biginds, dims=3), 1)
    Ξes[ips .< tol] .= 0
    Ξes = Ξes[:,:,1]
    smallinds = .!(abs.(Ξes) .> 0)
    for ind=1:size(y,2)
        biginds = .!smallinds[:,ind]
        any(biginds) && (Ξes[biginds,ind] = θ[:,biginds] \ y[:,ind])
    end
    return Ξes
end

function main(sys::Int, noise_idx::Int)
    @assert 1 <= sys <= 6
    NoisePct = NOISE_WG[noise_idx]
    fname, sysname, n_state, n_lib, dt, Tdefault = SYSTEMS[sys]
    Lib = LIBS[sys]
    data = load(datadir("sims", "ode_data", fname))

    println("=== WAIC/gMDL (corrected) rerun: $sysname, noise=$NoisePct% ===")
    flush(stdout)

    Ξs_raw = Dict(ic => zeros(n_lib, n_state, RUNS_WG) for ic in NEW_ICS)
    Ξs_eff = Dict(ic => zeros(n_lib, n_state, RUNS_WG) for ic in NEW_ICS)
    n_eff_run = zeros(RUNS_WG)

    Random.seed!(9000 + sys*100 + noise_idx)

    for run = 1:RUNS_WG
        println("  run $run/$RUNS_WG"); flush(stdout)
        qt, θ, n_eff_val = GetInputsWithNeff(sys, data["Xtrues"], data["ts"], Lib, NoisePct)
        n_eff_run[run] = n_eff_val
        for ic in NEW_ICS
            Ξr = EnAdSR_ext(θ, qt', ic; tol=TOL_WG, c=CC_WG, num_batches=NUM_BATCHES_WG)
            Ξs_raw[ic][:, :, run] = Ξr
            Ξe = EnAdSR_ext(θ, qt', ic; tol=TOL_WG, c=CC_WG, num_batches=NUM_BATCHES_WG, n_eff=n_eff_val, score_on_all=true)
            Ξs_eff[ic][:, :, run] = Ξe
        end
    end

    outdir = datadir("sims", "ode_results_rev", "waic_gmdl_neffall")
    mkpath(outdir)
    outfile = joinpath(outdir, "$(lowercase(replace(sysname, " " => "_")))_waic_gmdl_noise$(noise_idx).jld")
    out = Dict{String,Any}("sysname"=>sysname, "sys"=>sys, "NoisePct"=>NoisePct,
                            "Xitrue"=>data["Ξtrue"], "n_eff"=>n_eff_run)
    for ic in NEW_ICS
        out["Xis_raw_$ic"] = Ξs_raw[ic]
        out["Xis_eff_$ic"] = Ξs_eff[ic]
    end
    wsave(outfile, out)
    println("Wrote $outfile"); flush(stdout)
end

sys, noise_idx = if haskey(ENV, "SLURM_ARRAY_TASK_ID")
    t = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    (t ÷ 6) + 1, (t % 6) + 1
elseif length(ARGS) >= 2
    parse(Int, ARGS[1]), parse(Int, ARGS[2])
else
    error("Usage: julia rerun_with_waic_gmdl.jl SYS NOISE_IDX  (or SLURM_ARRAY_TASK_ID ∈ [0,35])")
end

main(sys, noise_idx)
