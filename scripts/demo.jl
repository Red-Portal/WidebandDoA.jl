
using Accessors
using Distributions
using HDF5
using LinearAlgebra
using MKL
using Plots, StatsPlots
using Random, Random123
using ReversibleJump
using StableRNGs
using Tullio
using WidebandDoA

include("common.jl")

function main()
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    rng  = Random123.Philox4x(UInt64, seed, 8)
    Random123.set_counter!(rng, 2)
    Random.seed!(rand(rng, UInt64))

    ϕ    = [30., 45, -15, -60]/180*π
    snrs = [-6.0, 4.0, 0.0, -4.0]

    N     = 64
    n_dft = 4*N
    fs    = 3000.0
    model = construct_default_model(N, fs)
    c, Δx = model.likelihood.c, model.likelihood.Δx
    y, _  = simulate_signal(rng, N, n_dft, ϕ, snrs, 100, 1000, fs, 1.0, Δx, c; visualize=true)
    y    /= std(y)
    cond  = WidebandConditioned(model, y)

    n_samples = 10^5

    prop  = UniformNormalLocalProposal(0.0, 2.0)
    mcmc  = SliceSteppingOut([2.0, 2.0])

    jump   = IndepJumpProposal(prop)

    initial_params = WidebandDoA.WidebandIsoIsoParam{Float64}[]
    initial_order  = 0
	
    rjmcmc = ReversibleJump.NonReversibleJumpMCMC(jump, mcmc; jump_rate=0.9)

    samples, stats = ReversibleJump.sample(
        rjmcmc,
        cond,
        n_samples,
        initial_order,
        initial_params;
        show_progress=true,
    )

    Plots.plot(
        [stat.order for stat in stats],
        xlabel="RJMCMC Iteration",
        ylabel="Model order",
        label="Model order trace"
    ) |> display

    samples_burned = samples[n_samples ÷ 10:end]
    stats_burned = stats[n_samples ÷ 10:end]

    k_post = [stat.order for stat in stats_burned]
    ϕ_post = [[target.phi for target in sample] for sample in samples_burned]

    k_mixture = round(Int, quantile(k_post, 0.8))
    mixture, labels = WidebandDoA.relabel(Random.default_rng(), ϕ_post, k_mixture; show_progress=false)

    relabeled = [ eltype(first(samples_burned))[] for _ in 1:k_mixture+1 ]
    for (i, sample) in enumerate(samples_burned)
        for (j, target) in enumerate(sample)
            push!(relabeled[labels[i][j]], target)
        end
    end

    Plots.plot() |> display
    for comp in first(relabeled, length(relabeled) - 1)
        Plots.histogram!([sample.phi for sample in comp], normed=true) |> display
    end

    # Plots.plot() |> display
    # for comp in first(relabeled, length(relabeled) - 1)
    #     Plots.histogram!([exp(sample.loglambda) for sample in comp], normed=true) |> display
    # end
    
    h5open(datadir("pro", "demo.h5"), "w") do io
        x_angles = range(-π/2, π/2; length=2^18) |> collect

        doa_structs = filter(sample -> length(sample) == length(ϕ), samples_burned)
        doas        = [doa_struct.phi for doa_struct in  vcat(doa_structs...) ]

        y_mixture_pdf = map(Base.Fix1(pdf, mixture), x_angles)
        y_comp1_pdf   = map(Base.Fix1(pdf, mixture.components[1]), x_angles)
        y_comp2_pdf   = map(Base.Fix1(pdf, mixture.components[2]), x_angles)
        y_comp3_pdf   = map(Base.Fix1(pdf, mixture.components[3]), x_angles)
        y_comp4_pdf   = map(Base.Fix1(pdf, mixture.components[4]), x_angles)

        loglambda_comp1 = [sample.loglambda for sample in relabeled[1]]
        loglambda_comp2 = [sample.loglambda for sample in relabeled[2]]
        loglambda_comp3 = [sample.loglambda for sample in relabeled[3]]
        loglambda_comp4 = [sample.loglambda for sample in relabeled[4]]

        order_post = [stat.order for stat in stats]

        write(io, "doas_posterior", doas)
        write(io, "doas_true",      ϕ)
        write(io, "x_angles",       x_angles)
        write(io, "y_mixture_pdf",  y_mixture_pdf)
        write(io, "y_comp1_pdf",    y_comp1_pdf)
        write(io, "y_comp2_pdf",    y_comp2_pdf)
        write(io, "y_comp3_pdf",    y_comp3_pdf)
        write(io, "y_comp4_pdf",    y_comp4_pdf)
        write(io, "k_post",         order_post)

        write(io, "loglamdba_true",  log.(10.0.^(snrs/10)))
        write(io, "loglambda_comp1", loglambda_comp1)
        write(io, "loglambda_comp2", loglambda_comp2)
        write(io, "loglambda_comp3", loglambda_comp3)
        write(io, "loglambda_comp4", loglambda_comp4)
    end
end
