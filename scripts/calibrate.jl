
using Accessors
using DataFrames, DataFramesMeta
using Plots, StatsPlots
using ProgressMeter
using Random, Random123
using JLD2

include("common.jl")


function run_rjmcmc(rng, cond, n_samples, n_burn)
    initial_params = WidebandDoA.WidebandIsoIsoParam{Float64}[]
    initial_order  = 0

    prop   = UniformNormalLocalProposal(0.0, 1.0)
    mcmc   = SliceSteppingOut([2.0, 2.0])
    jump   = IndepJumpProposal(prop)
    rjmcmc = ReversibleJump.NonReversibleJumpMCMC(jump, mcmc; jump_rate=0.9)

    _, stats = ReversibleJump.sample(
        rng,
        rjmcmc,
        cond,
        n_samples,
        initial_order,
        initial_params;
        show_progress=false,
    )
    stats     = last(stats, n_samples - n_burn)
    k_post    = modelposterior_naive(stats)
    k_post_rb = ReversibleJump.modelposterior(stats, cond.model.prior.order_prior)
    k_post, k_post_rb
end

function estimate_error(
    snr, ϕ, f_begin, f_end, fs, source_prior, n_samples, n_burn, n_reps
)
    k_true = length(ϕ)
    data   = pmap(1:n_reps) do key
        seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
        rng  = Philox4x(UInt64, seed, 8)
        set_counter!(rng, key)

	N     = 128
        n_dft = 1024
	model = construct_default_model(N, fs)
	c, Δx = model.likelihood.c, model.likelihood.Δx
	y, _  = simulate_signal(rng, N, n_dft, ϕ, snr, f_begin, f_end, fs, 1.0, Δx, c)
	cond  = WidebandConditioned(model, y)
        cond  = @set cond.model.prior.source_prior = source_prior

        k_post, k_post_rb = run_rjmcmc(rng, cond, n_samples, n_burn)   
        (
            zeroone_naive        = mode(k_post)    != k_true,
            zeroone_raoblackwell = mode(k_post_rb) != k_true,
            l1_naive             = abs(median(k_post)    - k_true),
            l1_raoblackwell      = abs(median(k_post_rb) - k_true),
        )
    end
    reduce_namedtuples(
        arr -> run_bootstrap(arr; confint_strategy=BCaConfInt(0.95)), data
    )
end

function run_simulation()
    n_samples = 2^12
    n_burn    = 2^7
    n_reps    = 2^7

    name    = "fullband"
    ϕ       = [-0.8, -0.4, 0.0, 0.4, 0.8]
    fs      = 2000.0
    f_begin = 0.0
    f_end   = fs/2

    # name    = "bandlimited"
    # ϕ       = [-0.8, -0.4, 0.0, 0.4, 0.8]
    # fs      = 2000.0
    # f_begin = [200,300,400,500,600]
    # f_end   = [300,400,500,600,700]

    prior = [
        (dist="inversegamma", param1=0.1,   param2=0.1),
        (dist="inversegamma", param1=0.01,  param2=0.01),
        (dist="inversegamma", param1=0.001, param2=0.001),
        (dist="uniform",      param1=0.1,   param2=10.0),
        (dist="uniform",      param1=0.01,  param2=100.0),
        (dist="uniform",      param1=0.5,   param2=5.0),
        (dist="lognormal",    param1=1.3,   param2=1.2),
        (dist="lognormal",    param1=5.3,   param2=2.3),
        (dist="lognormal",    param1=-0.8,  param2=0.6),
        (dist="lognormal",    param1=1.5,   param2=0.6),
    ]

    snrs = [-10, -8., -6, -4., -2, 0., 2, 4., 6, 8., 10]
    snrs = [(snr=snr,) for snr in snrs]

    configs = Iterators.product(prior, snrs) |> collect
    configs = reshape(configs, :)
    configs = map(x -> merge(x...), configs)

    df = @showprogress mapreduce(vcat, configs) do config
        (; dist, param1, param2, snr) = config

        source_prior = if dist == "lognormal"
            LogNormal(param1, param2)
        elseif dist == "inversegamma"
            InverseGamma(param1, param2)
        else
            Uniform(param1, param2)
        end

        res = estimate_error(
            snr, ϕ, f_begin, f_end, fs,
            source_prior, n_samples, n_burn, n_reps
        )

        df = DataFrame(
            dist   = dist,
            param1 = param1,
            param2 = param2,
            snr    = snr,
            #
            zeroone_naive_mean   = res.zeroone_naive[1],
            zeroone_naive_lower  = res.zeroone_naive[2],
            zeroone_naive_upper  = res.zeroone_naive[3],
            #
            zeroone_raoblackwell_mean  = res.zeroone_raoblackwell[1],
            zeroone_raoblackwell_lower = res.zeroone_raoblackwell[2],
            zeroone_raoblackwell_upper = res.zeroone_raoblackwell[3],
            #
            l1_naive_mean   = res.l1_naive[1],
            l1_naive_lower  = res.l1_naive[2],
            l1_naive_upper  = res.l1_naive[3],
            #
            l1_raoblackwell_mean  = res.l1_raoblackwell[1],
            l1_raoblackwell_lower = res.l1_raoblackwell[2],
            l1_raoblackwell_upper = res.l1_raoblackwell[3],
        )
        display(df)
        df
    end
    save(datadir("raw", "calibration_error_$(name).jld2"), "data", df) 
end

function process_data()
    df = JLD2.load(datadir("raw", "calibration_error_fullband.jld2"), "data")
    display(df)
    Plots.plot() |> display
    
    for (distname, param1, param2, dist) in [
        ("lognormal",    1.3,   1.2,  LogNormal(1.3, 1.2)),
        ("lognormal",    5.3,   2.3,  LogNormal(5.3, 2.3)),
        #("lognormal"   , -0.8,  0.6,  LogNormal(-0.8, 0.6)),
        #("lognormal"   ,  1.5,  0.6,  LogNormal(1.5, 0.6)),
        ("inversegamma",  0.01, 0.01, InverseGamma(0.01, 0.01)),
        #("inversegamma", 0.001, 0.001, InverseGamma(0.001, 0.001)),
        #("uniform"   ,  0.1,  10.0, Uniform(0.1,   10.0)),
        #("uniform"   , 0.01, 100.0, Uniform(0.01, 100.0)),
        #("uniform"   ,  0.5,   5.0, Uniform(0.5,    5.0)),
    ]
        res = @chain df begin
            @subset(
                :dist   .== distname,
                :param1 .== param1,
                :param2 .== param2
            )
            @orderby(:snr)
            @select(
                :l1_naive_mean,
                :l1_naive_lower,
                :l1_naive_upper
            )
            Array
        end
        display(res)
        xrange = -10:2:10
        Plots.plot!(
            xrange, res[:,1],
            ribbon=(abs.(res[:,2]), res[:,3]),
            label="$(distname)($(param1), $(param2))"
        ) |> display
        #Plots.plot!(xrange, x -> 5*pdf(dist, 10^(x/10))) |> display
    end
end
