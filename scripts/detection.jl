
using AbstractMCMC
using Base.Iterators
using DataFrames, DataFramesMeta
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
    if ENV["TASK"] == "null"
            name  = "detection_null.jld2"
            setup = (
                n_bins = 32,
                fs     = 3000.0,
            )
            @info(name, setup...)
            df = DataFrame()
            for snr in -14.:1.:10, n_snap in 2:2:16
                df_rjmcmc    = run_experiment(:rjmcmc,    setup.n_bins, n_snap, Float64[], snr, [], [], setup.fs)
                df_likeratio = run_experiment(:likeratio, setup.n_bins, n_snap, Float64[], snr, [], [], setup.fs)
                df_rjmcmc[   !, :snr]   .= snr
                df_likeratio[!, :snr]   .= snr
                df_rjmcmc[   !, :nsnap] .= n_snap
                df_likeratio[!, :nsnap] .= n_snap
                df′ = vcat(df_rjmcmc, df_likeratio)
                println(df′)
                df = vcat(df, df′)
                JLD2.save(datadir("raw", name), "data", df, "setup", setup)
            end

    elseif ENV["TASK"] == "wideband"
        for k in [2, 4, 6]
            name  = "detection_wideband_k=$(k)_varying_snr.jld2"
            setup = (
                n_bins  = 32,
                ϕ       = range(-2/6*π, 2/6*π; length=k),
                f_begin = 10.0,
                f_end   = 1000.0,
                fs      = 3000.0,
            )
            @info(name, setup...)
            df = DataFrame()
            for snr in -14.:1.:10, n_snap in 2:2:16
                df_rjmcmc    = run_experiment(:rjmcmc,    setup.n_bins, n_snap, setup.ϕ, snr, setup.f_begin, setup.f_end, setup.fs)
                df_likeratio = run_experiment(:likeratio, setup.n_bins, n_snap, setup.ϕ, snr, setup.f_begin, setup.f_end, setup.fs)
                df_rjmcmc[   !, :snr]   .= snr
                df_likeratio[!, :snr]   .= snr
                df_rjmcmc[   !, :nsnap] .= n_snap
                df_likeratio[!, :nsnap] .= n_snap
                df′ = vcat(df_rjmcmc, df_likeratio)
                println(df′)
                df = vcat(df, df′)
                JLD2.save(datadir("raw", name), "data", df, "setup", setup)
            end
        end

    elseif ENV["TASK"] == "narrowband"
        for k in [2, 4, 6]
            name  = "detection_narrowband_k=$(k).jld2"
            setup = (
                n_bins  = 32,
                ϕ       = range(-2/6*π, 2/6*π; length=k),
                f_begin = 400.0,
                f_end   = 500.0,
                fs      = 3000.0,
            )
            @info(name, setup...)
            df = DataFrame()
            for snr in -14.:1.:10, n_snap in 2:2:12
                df_rjmcmc    = run_experiment(:rjmcmc,    setup.n_bins, n_snap, setup.ϕ, snr, setup.f_begin, setup.f_end, setup.fs)
                df_likeratio = run_experiment(:likeratio, setup.n_bins, n_snap, setup.ϕ, snr, setup.f_begin, setup.f_end, setup.fs)
                df_rjmcmc[!,    :snr]   .= snr
                df_likeratio[!, :snr]   .= snr
                df_rjmcmc[   !, :nsnap] .= n_snap
                df_likeratio[!, :nsnap] .= n_snap
                df′ = vcat(df_rjmcmc, df_likeratio)
                println(df′)
                df = vcat(df, df′)
                JLD2.save(datadir("raw", name), "data", df, "setup", setup)
            end
        end

    elseif ENV["TASK"] == "mixedband"
        for k in [2, 4, 6]
            name  = "detection_mixedband_k=$(k).jld2"
            setup = (
                n_bins  = 32,
                ϕ       = range(-2/6*π, 2/6*π; length=k),
                f_begin = vcat(fill(400.0, k÷2), fill(  10.0, k÷2)),
                f_end   = vcat(fill(500.0, k÷2), fill(1000.0, k÷2)),
                fs      = 3000.0,
            )
            @info(name, setup...)
            df = DataFrame()
            for snr in -14.:1.:10, n_snap in 2:2:16
                df_rjmcmc    = run_experiment(:rjmcmc,    setup.n_bins, n_snap, setup.ϕ, snr, setup.f_begin, setup.f_end, setup.fs)
                df_likeratio = run_experiment(:likeratio, setup.n_bins, n_snap, setup.ϕ, snr, setup.f_begin, setup.f_end, setup.fs)
                df_rjmcmc[!,    :snr]   .= snr
                df_likeratio[!, :snr]   .= snr
                df_rjmcmc[   !, :nsnap] .= n_snap
                df_likeratio[!, :nsnap] .= n_snap
                df′ = vcat(df_rjmcmc, df_likeratio)
                println(df′)
                df = vcat(df, df′)
                JLD2.save(datadir("raw", name), "data", df, "setup", setup)
            end
        end

    elseif ENV["TASK"] == "separationunequal"
        name  = "detection_unequal_power_separation.jld2"
        setup = (
            n_bins   = 32,
            k        = 2,
            base_snr = 0.0,
            f_begin  = [100.0, 100.0,],
            f_end    = [500.0, 500.0,],
            fs       = 3000.0,
        )
        @info(name, setup...)
        df = DataFrame()
        for snr_diff in [0, 5, 10], separation in (1:2:20)*π/180, n_snap in 2:2:16
            ϕ   = [0.0, separation]
            snr = [setup.base_snr + snr_diff, setup.base_snr]
            
            df_rjmcmc    = run_experiment(:rjmcmc,    setup.n_bins, n_snap, ϕ, snr, setup.f_begin, setup.f_end, setup.fs)
            df_likeratio = run_experiment(:likeratio, setup.n_bins, n_snap, ϕ, snr, setup.f_begin, setup.f_end, setup.fs)

            df_rjmcmc[!,    :snr_diff]   .= snr_diff
            df_likeratio[!, :snr_diff]   .= snr_diff
            df_rjmcmc[!,    :separation] .= separation
            df_likeratio[!, :separation] .= separation
            df_rjmcmc[!,    :nsnap]      .= n_snap
            df_likeratio[!, :nsnap]      .= n_snap

            df′ = vcat(df_rjmcmc, df_likeratio)
            println(df′)
            df = vcat(df, df′)
            JLD2.save(datadir("raw", name), "data", df, "setup", setup)
        end
    end
end

function statistics(df, group_key, statistic)
    upper_conf(tup) = tup[1] + tup[2]
    lower_conf(tup) = tup[1] + tup[3]

    df = @chain groupby(df, group_key) begin
        @combine($"$(statistic)_boot" = run_bootstrap($statistic),)
        @transform(
            $"$(statistic)_mean"  = first.($"$(statistic)_boot"),
            $"$(statistic)_lower" = upper_conf.($"$(statistic)_boot"),
            $"$(statistic)_upper" = lower_conf.($"$(statistic)_boot"),
        )
        @orderby($group_key)
    end
end

function process_data()
    k    = 6
    name = "detection_narrowband_k=$(k)_varying_snr.jld2"

    df, setup = JLD2.load(datadir("raw", name), "data", "setup")
    @info("setup", setup...)

    df_rjmcmc    = statistics(@subset(df, :method .== Symbol("rjmcmc")),    :snr, :l1)
    df_likeratio = statistics(@subset(df, :method .== Symbol("likeratio")), :snr, :l1)

    Plots.plot(    df_rjmcmc.snr,    df_rjmcmc.l1_mean, ribbon=(   abs.(df_rjmcmc.l1_lower - df_rjmcmc.l1_mean),       df_rjmcmc.l1_upper - df_rjmcmc.l1_mean))
    Plots.plot!(df_likeratio.snr, df_likeratio.l1_mean, ribbon=(abs.(df_likeratio.l1_lower - df_likeratio.l1_mean), df_likeratio.l1_upper - df_likeratio.l1_mean))
end
