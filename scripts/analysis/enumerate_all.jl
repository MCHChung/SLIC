using DrWatson
@quickactivate "SLIC"
using JLD, Random, Statistics, LinearAlgebra

include(scriptsdir("analysis", "common.jl"))

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


# ============================================================================
# Exhaustive enumeration of candidate supports (Supplementary Fig. S12).
#
# The system is chosen with SLIC_ENUM_SYS; SLURM_ARRAY_TASK_ID selects the
# noise level and equation, task = (noise_idx-1)*n_state + (eq_idx-1), so there
# are 6 noise levels × n_state equations = 6*n_state tasks per system.
#
# max_k cap is per-system: full enumeration where tractable, max_k=8 for
# Lorenz/Rossler (true supports ≤3 per equation, so max_k=8 is effectively
# exhaustive).
# ============================================================================

const NOISE_ENUM = [0, 5, 10, 20, 30, 40]
const RUNS_ENUM = 25
const ICS_ENUM = ["slic", "aic", "aicc", "bic", "hqic", "bc", "kic"]
# Per-system conditioning weight, matching discover_model_ode_main.jl call sites.
# Spans four orders of magnitude, so it is set per system.
#   Lorenz=1e-1, Rossler=1e-1, LV=1e-3, Brusselator=1e-2, VdP=1e0, NLP=1e-2
const C_VALS = Dict(1 => 1e-1, 2 => 1e-1, 3 => 1e-3, 4 => 1e-2, 5 => 1e0, 6 => 1e-2)
# Sensitivity knob: multiply all c values by SLIC_C_SCALE (default 1.0). Lets us
# probe robustness of the enumeration to the conditioning floor without editing
# the per-system c. E.g. SLIC_C_SCALE=0.1 drops every c by an order of magnitude.
const C_SCALE = haskey(ENV, "SLIC_C_SCALE") ? parse(Float64, ENV["SLIC_C_SCALE"]) : 1.0
# Absolute override: if SLIC_C_ABS is set, it REPLACES C_VALS[sys] entirely
# (C_SCALE is then ignored). Used to run the noise sweep at the value calibrated
# on noise-free data by scripts/analysis/calibrate_c.jl.
const C_ABS = haskey(ENV, "SLIC_C_ABS") ? parse(Float64, ENV["SLIC_C_ABS"]) : nothing
c_for(sys) = C_ABS === nothing ? C_SCALE * C_VALS[sys] : C_ABS

# Dimensionless per-equation floor: if SLIC_GAMMA is set, use
#     η_eq = γ · RSS⁽⁰⁾_eq
# where RSS⁽⁰⁾ is the noise-free residual of the true model for that equation,
# read from the calibration written by scripts/analysis/calibrate_c.jl. This makes
# the floor scale with each equation's own systematic error rather than sharing
# one absolute value across equations whose residuals differ by orders of
# magnitude. cond(θ) plays no role in this mode.
const GAMMA = haskey(ENV, "SLIC_GAMMA") ? parse(Float64, ENV["SLIC_GAMMA"]) : nothing

function rss0_for(sysname_safe::String)
    p = datadir("sims", "ode_results", "c_calibration", "$(sysname_safe)_ccal.jld")
    isfile(p) || error("SLIC_GAMMA set but calibration missing: $p\n" *
                       "Run scripts/analysis/calibrate_c.jl first (SLIC_CAL_SYS=1-6).")
    return load(p)["rss_true"]
end

# Per-system: (sys_id, max_k cap)  (nothing = no cap, full 2^p)
const SYS_MAX_K = Dict(
    1 => 8,         # Lorenz: 19-term library, cap at 8 (saves ~12×)
    2 => 8,         # Rossler: 20-term library, cap at 8 (saves ~4×)
    3 => nothing,   # LV: 9-term library, full enumeration trivial
    4 => nothing,   # Brusselator: 10-term library, full enumeration trivial
    5 => nothing,   # VdP: full enumeration
    6 => nothing,   # Pendulum: full enumeration
)

function main(sys::Int, noise_idx::Int, eq_idx::Int)
    @assert haskey(SYSTEMS, sys)
    @assert 1 <= noise_idx <= length(NOISE_ENUM)
    NoisePct = NOISE_ENUM[noise_idx]

    fname, sysname, n_state, n_lib, dt, Tdefault = SYSTEMS[sys]
    @assert 1 <= eq_idx <= n_state
    Lib = LIBS[sys]
    max_k = SYS_MAX_K[sys]

    data = load(datadir("sims", "ode_data", fname))
    Ξtrue = data["Ξtrue"]

    n_total = max_k === nothing ? 2^n_lib : sum(binomial(n_lib, k) for k = 1:max_k)
    println("=== Enumeration: $sysname (sys=$sys), noise=$NoisePct%, eq=$eq_idx ===")
    println("    library=$n_lib, max_k=$(max_k === nothing ? "full" : max_k), supports=$n_total")
    flush(stdout)

    rng = Random.default_rng()
    Random.seed!(rng, 5000 + sys * 1000 + noise_idx * 10 + eq_idx)

    Ξs_raw_eq = Dict(ic => zeros(n_lib, RUNS_ENUM) for ic in ICS_ENUM)
    Ξs_eff_eq = Dict(ic => zeros(n_lib, RUNS_ENUM) for ic in ICS_ENUM)
    n_eff_run = zeros(RUNS_ENUM)

    for run = 1:RUNS_ENUM
        print("  run $run/$RUNS_ENUM ... ")
        flush(stdout)
        t0 = time()

        qt, θ, n_eff_val = GetInputsWithNeff(sys, data["Xtrues"], data["ts"], Lib, NoisePct)
        n_eff_run[run] = n_eff_val

        y_eq = reshape(qt'[:, eq_idx], (size(qt, 2), 1))

        # Conditioning term matching the main pipeline (η = c·cond(θ)) with the
        # SAME per-system c as discover_model_ode_main.jl; prevents competitors
        # chasing sub-noise-floor RSS reductions and over-selecting at zero noise.
        # Applied identically to all criteria including SLIC.
        sysname_safe0 = lowercase(replace(sysname, " " => "_"))
        η_cond = GAMMA === nothing ? c_for(sys) * cond(θ) :
                                     GAMMA * rss0_for(sysname_safe0)[eq_idx]

        Ξ_raw_dict, _, _ = enumerate_supports(θ, y_eq, ICS_ENUM;
                                                n_eff=nothing, max_k=max_k,
                                                η=η_cond, verbose=false)
        Ξ_eff_dict, _, _ = enumerate_supports(θ, y_eq, ICS_ENUM;
                                                n_eff=n_eff_val, max_k=max_k,
                                                η=η_cond, verbose=false)
        for ic in ICS_ENUM
            Ξs_raw_eq[ic][:, run] = Ξ_raw_dict[ic][:, 1]
            Ξs_eff_eq[ic][:, run] = Ξ_eff_dict[ic][:, 1]
        end

        elapsed = time() - t0
        println("done ($(round(elapsed/60, digits=2)) min)")
        flush(stdout)
    end

    sysname_safe = lowercase(replace(sysname, " " => "_"))
    cscale_tag = GAMMA !== nothing ? "_g" * string(GAMMA) :
                 C_ABS !== nothing ? "_ccal" :
                 (C_SCALE == 1.0 ? "" : "_cx" * string(C_SCALE))
    outdir = datadir("sims", "ode_results", "enumeration" * cscale_tag, sysname_safe)
    mkpath(outdir)
    outfile = joinpath(outdir, "$(sysname_safe)_enum_noise$(noise_idx)_eq$(eq_idx).jld")

    wsave(outfile, Dict{String, Any}(
        "sysname"     => sysname,
        "sys"         => sys,
        "NoisePct"    => NoisePct,
        "eq_idx"      => eq_idx,
        "max_k"       => max_k === nothing ? -1 : max_k,
        "n_supports"  => n_total,
        "Xitrue_col"  => Ξtrue[:, eq_idx],
        "n_eff"       => n_eff_run,
        "Xis_raw_eq"  => Ξs_raw_eq,
        "Xis_eff_eq"  => Ξs_eff_eq,
    ))
    println("Wrote $outfile")
    flush(stdout)
end

# ============================================================================
# Entry: select task via env var SLIC_ENUM_SYS, SLURM_ARRAY_TASK_ID
#
# The caller sets SLIC_ENUM_SYS=<sys_id>, and SLURM_ARRAY_TASK_ID
# encodes (noise_idx-1)*n_state + (eq_idx-1).
# ============================================================================

sys = parse(Int, ENV["SLIC_ENUM_SYS"])
n_state = SYSTEMS[sys][3]

task = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
noise_idx = (task ÷ n_state) + 1
eq_idx = (task % n_state) + 1

main(sys, noise_idx, eq_idx)
