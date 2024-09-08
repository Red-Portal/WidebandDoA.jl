
using AbstractMCMC
using Base.Iterators
using DataFrames, DataFramesMeta
using Distributed
using Distributions
using HDF5
using JLD2
using Plots
using ProgressMeter
using Random, Random123
using ReversibleJump
using WidebandDoA

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

function run_experiment(method, n_bins, n_snap, ϕ, snr, f_begin, f_end, fs; kwargs...)
    n_reps = 100
    seed   = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    k_true = length(ϕ)

    dfs = @showprogress pmap(1:n_reps) do key
        rng  = Random123.Philox4x(UInt64, seed, 8)
        Random123.set_counter!(rng, key)

        y = simulate_signal(rng, n_bins, n_snap, ϕ, snr, f_begin, f_end, fs)

        res = @timed begin
            if method == :rjmcmc
                estimate_rjmcmc(rng, n_bins, n_snap, fs, y; kwargs...)
            elseif method == :likeratio
                estimate_likeratiotest(rng, n_bins, n_snap, fs, y; kwargs...)
            else method == :dascfar
                estimate_subbanddascfar(rng, n_bins, n_snap, fs, y; kwargs...)
            end
        end
        k        = res.value
        t        = res.time
        l1       = abs(k - k_true)
        l0       = 1 - Int(k == k_true)
        pcorrect = Int(k == k_true)
        DataFrame(method=method, l1=l1, l0=l0, pcorrect=pcorrect, time=t)
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
            for n_snap in 1:1:32
                snr = 0.0
                df_rjmcmc    = run_experiment(:rjmcmc,    setup.n_bins, n_snap, Float64[], snr, [], [], setup.fs)
                df_likeratio = run_experiment(:likeratio, setup.n_bins, n_snap, Float64[], snr, [], [], setup.fs)
                df_rjmcmc[   !, :nsnap] .= n_snap
                df_likeratio[!, :nsnap] .= n_snap
                df′ = vcat(df_rjmcmc, df_likeratio)
                println(df′)
                df = vcat(df, df′)
                JLD2.save(datadir("raw", name), "data", df, "setup", setup)
            end

    elseif ENV["TASK"] == "wideband"
        for k in [2, 4, 6, 8, 10]
            name  = "detection_wideband_k=$(k).jld2"
            setup = (
                n_bins  = 32,
                ϕ       = range(-2/6*π, 2/6*π; length=k),
                f_begin = 10.0,
                f_end   = 1000.0,
                fs      = 3000.0,
            )
            @info(name, setup...)
            df = DataFrame()
            for n_snap in [1, 2, 4, 8, 16, 32], snr in -14.:1.:10
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
        for k in [2, 4, 6, 8, 10]
            name  = "detection_narrowband_k=$(k).jld2"
            setup = (
                n_bins  = 32,
                ϕ       = range(-2/6*π, 2/6*π; length=k),
                f_begin = 500.0,
                f_end   = 600.0,
                fs      = 3000.0,
            )
            @info(name, setup...)
            df = DataFrame()
            for n_snap in [1, 2, 4, 8, 16, 32], snr in -14.:1.:10
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
                f_begin = vcat(fill(500.0, k÷2), fill(  10.0, k÷2)),
                f_end   = vcat(fill(600.0, k÷2), fill(1000.0, k÷2)),
                fs      = 3000.0,
            )
            @info(name, setup...)
            df = DataFrame()
            for n_snap in [1, 2, 4, 8, 16, 32], snr in -14.:1.:10
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

    elseif ENV["TASK"] == "separation_wideband"
        name  = "detection_separation_wideband.jld2"
        setup = (
            n_bins   = 32,
            k        = 2,
            f_begin  = [  10.0,   10.0,],
            f_end    = [1000.0, 1000.0,],
            fs       = 3000.0,
        )
        @info(name, setup...)
        df = DataFrame()
        for snr_diff   in [0, 5,],
            n_snap     in [1, 2, 4, 8, 16, 32],
            base_snr   in -8:4:4,
            separation in (1:.5:10)*π/180

            ϕ   = [0.0, separation]
            snr = [base_snr + snr_diff, base_snr]

            df_rjmcmc    = run_experiment(:rjmcmc,    setup.n_bins, n_snap, ϕ, snr, setup.f_begin, setup.f_end, setup.fs)
            df_likeratio = run_experiment(:likeratio, setup.n_bins, n_snap, ϕ, snr, setup.f_begin, setup.f_end, setup.fs)

            df_rjmcmc[!,    :base_snr]   .= base_snr
            df_likeratio[!, :base_snr]   .= base_snr
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

    elseif ENV["TASK"] == "separation_narrowband"
        name  = "detection_separation_narrowband.jld2"
        setup = (
            n_bins   = 32,
            k        = 2,
            f_begin  = [500.0, 500.0,],
            f_end    = [600.0, 600.0,],
            fs       = 3000.0,
        )
        @info(name, setup...)
        df = DataFrame()
        for snr_diff   in [0, 5,],
            n_snap     in [1, 2, 4, 8, 16, 32],
            base_snr   in -8:4:4,
            separation in (1:.5:10)*π/180

            ϕ   = [0.0, separation]
            snr = [base_snr + snr_diff, base_snr]

            df_rjmcmc    = run_experiment(:rjmcmc,    setup.n_bins, n_snap, ϕ, snr, setup.f_begin, setup.f_end, setup.fs)
            df_likeratio = run_experiment(:likeratio, setup.n_bins, n_snap, ϕ, snr, setup.f_begin, setup.f_end, setup.fs)

            df_rjmcmc[!,    :base_snr]   .= base_snr
            df_likeratio[!, :base_snr]   .= base_snr
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

function convert_plot_series_pcorrect(df, xkey)
    pcorrect       = @. 1 - df.l0_mean
    pcorrect_upper = @. abs((1 - df.l0_lower) - pcorrect)
    pcorrect_lower = @. abs((1 - df.l0_upper) - pcorrect)

    x = df[:,xkey]
    y = hcat(pcorrect, pcorrect_upper, pcorrect_lower)' |> Array
    x, y
end

function process_data()
    # begin
    #     name = "detection_null"

    #     df, setup = JLD2.load(datadir("raw", name*".jld2"), "data", "setup")
    #     @info("setup", setup...)

    #     Plots.plot()

    #     h5open(datadir("pro", name*".h5"), "w") do io
    #         snr = 0.0

    #         df_rjmcmc    = statistics(
    #             @subset(df, :snr .== snr, :method .== Symbol("rjmcmc")),  :nsnap, :l0
    #         )
    #         df_likeratio = statistics(
    #             @subset(df, :snr .== snr, :method .== Symbol("likeratio")), :nsnap, :l0
    #         )

    #         println(df_rjmcmc)

    #         Plots.plot!(df_rjmcmc.nsnap ,   1 .- df_rjmcmc.l0_mean   , color=:blue) |> display
    #         Plots.plot!(df_likeratio.nsnap, 1 .- df_likeratio.l0_mean, color=:red ) |> display

    #         x_rjmcmc, y_rjmcmc       = convert_plot_series_pcorrect(df_rjmcmc,    :nsnap)
    #         x_likeratio, y_likeratio = convert_plot_series_pcorrect(df_likeratio, :nsnap)

    #         write(io, "x_rjmcmc",    x_rjmcmc)
    #         write(io, "y_rjmcmc",    y_rjmcmc)
    #         write(io, "x_likeratio", x_likeratio)
    #         write(io, "y_likeratio", y_likeratio)
    #     end
    # end

    # for name in [
    #     "detection_narrowband_k=2",
    #     "detection_wideband_k=2",
    #     "detection_mixedband_k=2",
    #     #
    #     "detection_narrowband_k=4",
    #     "detection_wideband_k=4",
    #     "detection_mixedband_k=4",
    #     #
    #     "detection_narrowband_k=6",
    #     "detection_wideband_k=6",
    #     "detection_mixedband_k=6",
    # ]
    #     df, setup = JLD2.load(datadir("raw", name*".jld2"), "data", "setup")
    #     @info("setup", setup...)

    #     Plots.plot()

    #     h5open(datadir("pro", name*".h5"), "w") do io
    #         for nsnap in [1, 2, 4, 8, 12]
    #             df_rjmcmc    = statistics(
    #                 @subset(df, :nsnap .== nsnap, :method .== Symbol("rjmcmc")),  :snr, :l0
    #             )
    #             df_likeratio = statistics(
    #                 @subset(df, :nsnap .== nsnap, :method .== Symbol("likeratio")), :snr, :l0
    #             )

    #             Plots.plot!(df_rjmcmc.snr   , 1 .- df_rjmcmc.l0_mean   , color=:blue) |> display
    #             Plots.plot!(df_likeratio.snr, 1 .- df_likeratio.l0_mean, color=:red ) |> display

    #             pcorrect_rjmcmc       = @. 1 - df_rjmcmc.l0_mean
    #             pcorrect_rjmcmc_upper = @. abs((1 - df_rjmcmc.l0_lower) - pcorrect_rjmcmc)
    #             pcorrect_rjmcmc_lower = @. abs((1 - df_rjmcmc.l0_upper) - pcorrect_rjmcmc)
    #             x_rjmcmc = df_rjmcmc.snr
    #             y_rjmcmc = hcat(
    #                 pcorrect_rjmcmc, pcorrect_rjmcmc_upper, pcorrect_rjmcmc_lower
    #             )' |> Array

    #             pcorrect_likeratio       = @. 1 - df_likeratio.l0_mean
    #             pcorrect_likeratio_upper = @. abs((1 - df_likeratio.l0_lower) - pcorrect_likeratio)
    #             pcorrect_likeratio_lower = @. abs((1 - df_likeratio.l0_upper) - pcorrect_likeratio)
    #             x_likeratio = df_rjmcmc.snr
    #             y_likeratio = hcat(
    #                 pcorrect_likeratio, pcorrect_likeratio_upper, pcorrect_likeratio_lower
    #             )' |> Array

    #             write(io, "x_rjmcmc_$(nsnap)", x_rjmcmc)
    #             write(io, "y_rjmcmc_$(nsnap)", y_rjmcmc)
    #             write(io, "x_likeratio_$(nsnap)", x_likeratio)
    #             write(io, "y_likeratio_$(nsnap)", y_likeratio)
    #         end
    #     end
    # end

    for name in [
        #"detection_separation_narrowband"
        "detection_separation_wideband"
    ]

        df, setup = JLD2.load(datadir("raw", name*".jld2"), "data", "setup")
        @info("setup", setup...)
        Plots.plot()

        begin
            nsnap    = 8
            base_snr = 4
            snr_diff = 0

            df_rjmcmc    = statistics(
                @subset(
                    df,
                    :base_snr .== base_snr,
                    :nsnap    .== nsnap,
                    :snr_diff .== snr_diff,
                    :method   .== Symbol("rjmcmc")
                ), :separation, :l0
            )
            df_likeratio = statistics(
                @subset(
                    df,
                    :base_snr .== base_snr,
                    :nsnap    .== nsnap,
                    :snr_diff .== snr_diff,
                    :method   .== Symbol("likeratio")
                ), :separation, :l0
            )

            display(df_rjmcmc)
            display(df_likeratio)

            Plots.plot!(
                df_rjmcmc.separation*180/π,
                1 .- df_rjmcmc.l0_mean,
                #ribbon=(
                #    @. abs.(df_rjmcmc.l0_lower + (1 - df_rjmcmc.l0_mean)),
                #    @. df_rjmcmc.l0_upper      + (1 - df_rjmcmc.l0_mean)
                #),
                color=:blue
            )  |> display
            Plots.plot!(
                df_likeratio.separation*180/π,
                1 .- df_likeratio.l0_mean,
                #ribbon=(
                #    @. abs(df_likeratio.l0_lower + (1 - df_likeratio.l0_mean)),
                #    @. df_likeratio.l0_upper     + (1 - df_likeratio.l0_mean)
                #),
                color=:red
            ) |> display
        end
    end
end
