
using Accessors
using DataFrames, DataFramesMeta
using Plots, StatsPlots
using ProgressMeter
using Random, Random123

include("common.jl")

function modelposterior_naive(stats)
    k_post = [stat.order for stat in stats]
    k_max  = maximum(k_post)
    sup    = 0:k_max+1
    counts = Dict{Int,Int}()

    for k in sup
        counts[k] = 0
    end

    for k in k_post
        counts[k] += 1
    end

    probs = zeros(length(sup))
    n     = length(k_post)
    for k in 0:k_max+1
        probs[k+1] = counts[k]/n
    end
    DiscreteNonParametric(sup, probs)
end

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

function estimate_error(snr, ϕ, source_prior, n_samples, n_burn, n_reps)
    k_true = length(ϕ)
    data   = pmap(1:n_reps) do key
        seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
        rng  = Philox4x(UInt64, seed, 8)
        set_counter!(rng, key)

        cond, _ = construct_default_model(rng, ϕ, snr)
        cond    = @set cond.model.prior.source_prior = source_prior

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
    ϕ         = [-4, -3., -2, -1, 1, 2, 3, 4]*π/9

    prior = [
        (dist="inversegamma", param1=0.1,   param2=0.1),
        (dist="inversegamma", param1=0.01,  param2=0.01),
        (dist="inversegamma", param1=0.001, param2=0.001),
        (dist="normal",       param1=1.3,   param2=1.2),
        (dist="normal",       param1=5.3,   param2=2.3),
        (dist="normal",       param1=-0.8,  param2=0.6),
        (dist="normal",       param1=1.5,   param2=0.6),
    ]

    snrs = [-10, -8., -6, -4., -2, 0., 2, 4., 6, 8., 10]
    snrs = [(snr=snr,) for snr in snrs]

    configs = Iterators.product(prior, snrs) |> collect
    configs = reshape(configs, :)
    configs = map(x -> merge(x...), configs)

    df = @showprogress mapreduce(vcat, configs) do config
        (; dist, param1, param2, snr) = config

        source_prior = if dist == "normal"
            Normal(param1, param2)
        else
            InverseGamma(param1, param2)
        end

        res = estimate_error(
            snr, ϕ, source_prior, n_samples, n_burn, n_reps
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
    save(datadir("raw", "calibration_error.jld2"), "data", df) 
end

function process_data()
    df = JLD2.load(datadir("raw", "calibration_error.jld2"), "data")
    display(df)
    
    res = @chain df begin
        @subset(
            :dist   .== "normal",
            :param1 .== 5.3,
            :param2 .== 2.3
        )
        @orderby(:snr)
        @select(:l1_naive_mean, :l1_naive_lower, :l1_naive_upper)
        Array
    end
    display(res)
    Plots.plot(-10:2:10, res[:,1], ribbon=(abs.(res[:,2]), res[:,3]))
end
