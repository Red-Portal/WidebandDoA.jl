
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
        show_progress=false,
    )
    stats = last(stats, n_samples - n_burn)
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
        model = @set model.prior = setproperties(model.prior, alpha_lambda=α_λ, beta_lambda=β_λ)
        k_post, k_post_rb = run_rjmcmc(rng, model, n_samples, n_burn)   
        (
            naive        = abs(mode(k_post)    - k_true),
            raoblackwell = abs(mode(k_post_rb) - k_true),
        )
    end
    reduce_namedtuples(
        arr -> run_bootstrap(arr; confint_strategy=BCaConfInt(0.95)), data
    )
end

function run_simulation()
    system_setup(; use_mkl=true, start=0)

    n_samples = 2^14
    n_burn    = 2^10
    n_reps    = 2^7
    ϕ         = [-4, -3., -2, -1, 1, 2, 3, 4]*π/9

    hypers = [
        # P[ 0.1 < λ < 10 ] ≈ 0.9
        (alpha_lambda = 2.01, beta_lambda = 0.40),
        (alpha_lambda = 2.01, beta_lambda = 5.84),

        (alpha_lambda = 3.01, beta_lambda = 0.55),
        (alpha_lambda = 3.01, beta_lambda = 11.64),

        (alpha_lambda = 4.01, beta_lambda = 0.68),
        (alpha_lambda = 4.01, beta_lambda = 18.12),

        # P[ 0.1 < λ < 10 ] ≈ 0.95
        (alpha_lambda = 2.01, beta_lambda = 0.49),
        (alpha_lambda = 2.01, beta_lambda = 3.96),

        (alpha_lambda = 3.01, beta_lambda = 0.64),
        (alpha_lambda = 3.01, beta_lambda = 8.69),

        (alpha_lambda = 4.01, beta_lambda = 0.79),
        (alpha_lambda = 4.01, beta_lambda = 14.25),

        # P[ 0.1 < λ < 10 ] ≈ 0.99
        (alpha_lambda = 2.01, beta_lambda = 0.70),
        (alpha_lambda = 2.01, beta_lambda = 1.72),

        (alpha_lambda = 3.01, beta_lambda = 0.86),
        (alpha_lambda = 3.01, beta_lambda = 4.71),

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
        (; naive, raoblackwell) = estimate_error(
            snr, ϕ, alpha_lambda, beta_lambda, n_samples, n_burn, n_reps
        )
        DataFrame(
            alpha_lambda = alpha_lambda,
            beta_lambda  = beta_lambda,
            snr          = snr,
            #
            naive_mean   = naive[1],
            naive_upper  = naive[2],
            naive_lower  = naive[3],
            #
            raoblackwell_mean  = raoblackwell[1],
            raoblackwell_upper = raoblackwell[2],
            raoblackwell_lower = raoblackwell[3],
        )
    end
    save(datadir("raw", "calibration_error.jld2"), "data", df) 
end
