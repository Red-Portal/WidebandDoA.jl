
using Accessors
using DataFrames
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

function run_rjmcmc(rng, model, n_samples, n_burn)
    initial_params = WidebandDoA.WidebandNormalGammaParam{Float64}[]
    initial_order  = 0

    prop   = UniformNormalLocalProposal(0.0, 1.0)
    mcmc   = SliceSteppingOut([2.0, 2.0])
    jump   = IndepJumpProposal(prop)
    rjmcmc = ReversibleJump.NonReversibleJumpMCMC(jump, mcmc; jump_rate=0.9)

    _, stats = ReversibleJump.sample(
        rng,
        rjmcmc,
        model,
        n_samples,
        initial_order,
        initial_params;
        show_progress=true,
    )
    stats     = last(stats, n_samples - n_burn)
    k_post    = modelposterior_naive(stats)
    k_post_rb = ReversibleJump.modelposterior(stats, model.prior.order_prior)
    k_post, k_post_rb
end

function estimate_error(snr, ϕ, α_λ, β_λ, n_samples, n_burn, n_reps)
    k_true = length(ϕ)
    data   = pmap(1:n_reps) do key
        seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
        rng  = Philox4x(UInt64, seed, 8)
        set_counter!(rng, key)

        model, _ = construct_default_model(rng, ϕ, snr)

        #y, a  = sample_bandlimited_signals(rng, model.prior, θ_true, 10, 300)
        #model = WidebandNormalGamma(y, model.prior)

        model = @set model.prior = setproperties(model.prior, alpha_lambda=α_λ, beta_lambda=β_λ)
        k_post, k_post_rb = run_rjmcmc(rng, model, n_samples, n_burn)   
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
    ϕ         = [-3., -2, -1, 1, 2, 3]*π/7

    hypers = [
        (alpha_lambda = 0.01, beta_lambda = 0.01),
        (alpha_lambda = 0.01, beta_lambda = 0.1),
        (alpha_lambda = 0.1,  beta_lambda = 0.01),
        (alpha_lambda = 0.1,  beta_lambda = 0.1),

        # P[ 0.1 < λ < 10 ] ≈ 0.9
        (alpha_lambda = 2.01, beta_lambda = 0.40),
        (alpha_lambda = 2.01, beta_lambda = 5.84),

        (alpha_lambda = 4.01, beta_lambda = 0.68),
        (alpha_lambda = 4.01, beta_lambda = 18.12),

        # P[ 0.1 < λ < 10 ] ≈ 0.95
        (alpha_lambda = 2.01, beta_lambda = 0.49),
        (alpha_lambda = 2.01, beta_lambda = 3.96),

        (alpha_lambda = 4.01, beta_lambda = 0.79),
        (alpha_lambda = 4.01, beta_lambda = 14.25),

        # P[ 0.1 < λ < 10 ] ≈ 0.99
        (alpha_lambda = 2.01, beta_lambda = 0.70),
        (alpha_lambda = 2.01, beta_lambda = 1.72),

        (alpha_lambda = 4.01, beta_lambda = 1.02),
        (alpha_lambda = 4.01, beta_lambda = 8.66),
    ]

    snrs = [-10, -8., -6, -4., -2, 0., 2, 4., 6, 8., 10]
    snrs = [(snr=snr,) for snr in snrs]

    configs = Iterators.product(hypers, snrs) |> collect
    configs = reshape(configs, :)
    configs = map(x -> merge(x...), configs)

    df = @showprogress mapreduce(vcat, configs) do config
        (; alpha_lambda, beta_lambda, snr) = config
        res = estimate_error(
            snr, ϕ, alpha_lambda, beta_lambda, n_samples, n_burn, n_reps
        )
        df = DataFrame(
            alpha_lambda = alpha_lambda,
            beta_lambda  = beta_lambda,
            snr          = snr,
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
