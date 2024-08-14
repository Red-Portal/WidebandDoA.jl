
using AbstractMCMC
using Base.Iterators
using DataFrames
using Distributions
using Distributed
using Random, Random123
using ReversibleJump
using WidebandDoA
using ProgressMeter
using Plots
using JLD2

include("common.jl")
include("baselines/baselines.jl")

function estimate_rjmcmc(
    rng   ::Random.AbstractRNG,
    n_bins::Int,
    n_snap::Int,
    fs    ::Real,
    y     ::AbstractMatrix
)
    n_samples = 2^12
    n_burn    = 2^10

    model = construct_default_model(n_bins*n_snap, fs)
    cond  = WidebandConditioned(model, y)

    prop = UniformNormalLocalProposal(0.0, 2.0)
    mcmc = SliceSteppingOut([2.0, 2.0])
    jump = IndepJumpProposal(prop)

    initial_params = WidebandDoA.WidebandIsoIsoParam{Float64}[]
    initial_order  = 0

    rjmcmc = ReversibleJump.NonReversibleJumpMCMC(jump, mcmc; jump_rate=0.9)

    _, stats = ReversibleJump.sample(
        rng,
        rjmcmc,
        cond,
        n_samples + n_burn,
        initial_order,
        initial_params;
        show_progress=false,
    )
    stats_burn = stats[n_burn+1:end]
    model_post = modelposterior_naive(stats_burn)
    median(model_post)
end

function estimate_likeratiotest(
    rng   ::Random.AbstractRNG,
    n_bins::Int,
    n_snap::Int,
    fs    ::Real,
    y     ::AbstractMatrix
)
    fdr   = 0.1 
    model = construct_default_model(n_bins*n_snap, fs)
    c     = model.likelihood.c
    Δx    = model.likelihood.Δx

    R, y, f_range = snapshot_covariance(y, n_bins, fs, n_snap)
    R_sel = R[:,:,2:end] 
    y_sel = y[:,:,2:end]
    f_sel = f_range[2:end] 

    k, _  = likeratiotest(
        rng,
        y_sel,
        R_sel,
        fdr,
        10,
        n_snap,
        f_sel,
        ArrayConfig(c, Δx);
        visualize=false,
    )
    k
end

function simulate_signal(rng, n_bins, n_snap, ϕ, snr, f_begin, f_end, fs)
    model = construct_default_model(1, fs)
    c, Δx = model.likelihood.c, model.likelihood.Δx
    n_dft = nextpow(2, n_snap*n_bins*2)
    y, _  = simulate_signal(
        rng, n_bins*n_snap, n_dft, ϕ, snr, f_begin, f_end, fs, 1.0, Δx, c
    )
    y
end

function run_experiment(method, n_bins, n_snap, ϕ, snr, f_begin, f_end, fs)
    n_reps = 32
    seed   = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    k_true = length(ϕ)

    dfs = @showprogress pmap(1:n_reps) do key
        rng  = Random123.Philox4x(UInt64, seed, 8)
        Random123.set_counter!(rng, key)

        y = simulate_signal(rng, n_bins, n_snap, ϕ, snr, f_begin, f_end, fs)

        k = if method == :rjmcmc
            estimate_rjmcmc(rng, n_bins, n_snap, fs, y)
        else
            estimate_likeratiotest(rng, n_bins, n_snap, fs, y)
        end
        l1 = abs(k - k_true)
        l0 = 1 - Int(k == k_true)
        DataFrame(method=method, l1=l1, l0=l0,)
    end
    vcat(dfs...)
end

function main()
    for k in [2, 4, 6, 8]
        name  = "wideband_k=$(k)_varying_snr.jld2"
        setup = (
            n_bins  = 32,
            n_snap  = 4,
            ϕ       = range(-2/6*π, 2/6*π; length=k),
            f_begin = 10.0,
            f_end   = 1000.0,
            fs      = 3000.0,
        )
        @info(name, setup...)
        df = DataFrame()
        for snr in -14.:1.:10
            df_rjmcmc    = run_experiment(:rjmcmc,    setup.n_bins, setup.n_snap, setup.ϕ, snr, setup.f_begin, setup.f_end, setup.fs)
            df_likeratio = run_experiment(:likeratio, setup.n_bins, setup.n_snap, setup.ϕ, snr, setup.f_begin, setup.f_end, setup.fs)
            df_rjmcmc[!, :snr]    .= snr
            df_likeratio[!, :snr] .= snr
            df′ = vcat(df_rjmcmc, df_likeratio)
            println(df′)
            df = vcat(df, df′)
        end
        JLD2.save(datadir("raw", name), "data", df, "setup", setup)
    end

    for k in [2, 4, 6, 8]
        name  = "narrowband_k=$(k)_varying_snr.jld2"
        setup = (
            n_bins  = 32,
            n_snap  = 4,
            ϕ       = range(-2/6*π, 2/6*π; length=k),
            f_begin = 400.0,
            f_end   = 500.0,
            fs      = 3000.0,
        )
        @info(name, setup...)
        df = DataFrame()
        for snr in -14.:1.:10
            df_rjmcmc    = run_experiment(:rjmcmc,    setup.n_bins, setup.n_snap, setup.ϕ, snr, setup.f_begin, setup.f_end, setup.fs)
            df_likeratio = run_experiment(:likeratio, setup.n_bins, setup.n_snap, setup.ϕ, snr, setup.f_begin, setup.f_end, setup.fs)
            df_rjmcmc[!, :snr]    .= snr
            df_likeratio[!, :snr] .= snr
            df′ = vcat(df_rjmcmc, df_likeratio)
            println(df′)
            df = vcat(df, df′)
        end
        JLD2.save(datadir("raw", name), "data", df, "setup", setup)
    end

    for k in [2, 4, 6, 8]
        name  = "mixedband_k=$(k)_varying_snr.jld2"
        setup = (
            n_bins  = 32,
            n_snap  = 4,
            ϕ       = range(-2/6*π, 2/6*π; length=k),
            f_begin = vcat(fill(400.0, k÷2), fill(k÷2,   10.0)),
            f_end   = vcat(fill(500.0, k÷2), fill(k÷2, 1000.0)),
            fs      = 3000.0,
        )
        @info(name, setup...)
        df = DataFrame()
        for snr in -14.:1.:10
            df_rjmcmc    = run_experiment(:rjmcmc,    setup.n_bins, setup.n_snap, setup.ϕ, snr, setup.f_begin, setup.f_end, setup.fs)
            df_likeratio = run_experiment(:likeratio, setup.n_bins, setup.n_snap, setup.ϕ, snr, setup.f_begin, setup.f_end, setup.fs)
            df_rjmcmc[!, :snr]    .= snr
            df_likeratio[!, :snr] .= snr
            df′ = vcat(df_rjmcmc, df_likeratio)
            println(df′)
            df = vcat(df, df′)
        end
        JLD2.save(datadir("raw", name), "data", df, "setup", setup)
    end

    begin
        k     = 4
        name  = "equalsignals_varying_snapshots.jld2"
        setup = (
            n_bins  = 32,
            k       = k,
            ϕ       = range(-2/6*π, 2/6*π; length=k),
            snr     = -4.0,
            f_begin = 10.0,
            f_end   = 500.0,
            fs      = 3000.0,
        )
        @info(name, setup...)
        df = DataFrame()
        for n_snap in 1:12
            df_rjmcmc    = run_experiment(:rjmcmc,    setup.n_bins, n_snap, setup.ϕ, setup.snr, setup.f_begin, setup.f_end, setup.fs)
            df_likeratio = run_experiment(:likeratio, setup.n_bins, n_snap, setup.ϕ, setup.snr, setup.f_begin, setup.f_end, setup.fs)
            df_rjmcmc[!, :n_snap]    .= n_snap
            df_likeratio[!, :n_snap] .= n_snap
            df′ = vcat(df_rjmcmc, df_likeratio)
            println(df′)
            df = vcat(df, df′)
        end
        JLD2.save(datadir("raw", name), "data", df, "setup", setup)
    end

    begin
        name  = "unequal_power_varying_separation.jld2"
        setup = (
            n_bins   = 32,
            n_snap   = 8,
            k        = 2,
            base_snr = 0.0,
            f_begin  = [100.0, 100.0,],
            f_end    = [500.0, 500.0,],
            fs       = 3000.0,
        )
        @info(name, setup...)
        df = DataFrame()
        for snr_diff in [0, 5, 10], separation in (1:20)*π/180
            ϕ   = [0.0, separation]
            snr = [setup.base_snr + snr_diff, setup.base_snr]

            df_rjmcmc    = run_experiment(:rjmcmc,    setup.n_bins, setup.n_snap, ϕ, snr, setup.f_begin, setup.f_end, setup.fs)
            df_likeratio = run_experiment(:likeratio, setup.n_bins, setup.n_snap, ϕ, snr, setup.f_begin, setup.f_end, setup.fs)

            df_rjmcmc[!,    :snr_diff]   .= snr_diff
            df_likeratio[!, :snr_diff]   .= snr_diff
            df_rjmcmc[!,    :separation] .= separation
            df_likeratio[!, :separation] .= separation

            df′ = vcat(df_rjmcmc, df_likeratio)
            println(df′)
            df = vcat(df, df′)
        end
        JLD2.save(datadir("raw", name), "data", df, "setup", setup)
    end
end
