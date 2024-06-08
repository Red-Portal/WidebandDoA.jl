
using AbstractMCMC
using Accessors
using Distributed
using Distributions
using DrWatson
using HDF5
using MCMCDiagnosticTools
using mcmcse
using ProgressMeter
using Random, Random123
using ReversibleJump
using WidebandDoA

@quickactivate

include("common.jl")

function reduce_namedtuples(f, vector_of_tuples)
    ks = keys(first(vector_of_tuples))
    @assert all(tup -> keys(tup) == ks, vector_of_tuples) 
    tuple_of_vectors = NamedTuple(k => getproperty.(vector_of_tuples, k) for k in ks)
    NamedTuple(k => f(v) for (k,v) in pairs(tuple_of_vectors))
end

function simulate_markov_chain(rng, mcmc, target, θ0, n_burn::Int, n_samples::Int)
    θ            = copy(θ0)
    stats_chain  = Vector{NamedTuple}(undef, n_samples)
    params_chain = Matrix{Float64}(   undef, 2, n_samples)

    for _ in 1:n_burn
        θ, _, _ = transition_mcmc(rng, mcmc, target, θ)
    end

    for t in 1:n_samples
        θ, _, stats    = transition_mcmc(rng, mcmc, target, θ)

        params_chain[1,t] = only(θ).phi
        params_chain[2,t] = exp(only(θ).loglambda)
        stats_chain[t]    = stats
    end

    mcmc_stats = reduce_namedtuples(mean, stats_chain)
    ess_stats  = (iact_phi    = mcmcse.mcvar(params_chain[1,:], r=2)/var(params_chain[1,:]),
                  iact_lambda = mcmcse.mcvar(params_chain[2,:], r=2)/var(params_chain[2,:]))
    merge(mcmc_stats, ess_stats)
end

function run_simulation(snr, mcmc, n_burn, n_samples, n_reps)
    stats = @showprogress pmap(1:n_reps) do key
        seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
        rng  = Philox4x(UInt64, seed, 8)
        set_counter!(rng, key)

        ϕ        = rand(rng, Uniform(-π/2, π/2))
        model, _ = construct_default_model(rng, [ϕ], snr)
        θ0       = [WidebandDoA.WidebandNormalGammaParam(ϕ, randn(rng))]

        simulate_markov_chain(rng, mcmc, model, θ0, n_burn, n_samples)
    end
    reduce_namedtuples(
        arr -> run_bootstrap(arr; confint_strategy=BCaConfInt(0.95)), stats
    )
end

function main()
    n_burn    = 128
    n_samples = 2^16
    n_reps    = 32

    for (mcmc, name) in [
        (
            WidebandNormalGammaMetropolis(
                MetropolisMixture(
                    IndependentMetropolis(Uniform(-π/2, π/2)),
                    RandomWalkMetropolis(0.1),
                ),
                MetropolisMixture(
                    IndependentMetropolis(Normal(0, 2)),
                    RandomWalkMetropolis(0.5),
                )
            ),
            "hybrid"
        ),
        (
            WidebandNormalGammaMetropolis(
                IndependentMetropolis(Uniform(-π/2, π/2)),
                IndependentMetropolis(Normal(0, 2))
            ),
            "imh"
        ),
        (
            WidebandNormalGammaMetropolis(
                RandomWalkMetropolis(0.1),
                RandomWalkMetropolis(1.5)
            ),
            "rwmh"
        ),
        (
            SliceSteppingOut([2.0, 2.0]),
            "slice"
        )
    ]
        for snr in [-8, -4, 0, 4]
            stats = run_simulation(snr, mcmc, n_burn, n_samples, n_reps)
            if name != "slice"
                @info(
                    "",
                    name,
                    snr,
                    phi_acceptance_rate       = round.(stats.phi_acceptance_rate,       digits=3),
                    loglambda_acceptance_rate = round.(stats.loglambda_acceptance_rate, digits=3),
                    iact_phi                  = round.(stats.iact_phi,                  digits=3),
                    iact_loglambda            = round.(stats.iact_lambda,               digits=3),
                )
            else
                @info(
                    "",
                    name,
                    snr,
                    iact_phi       = round.(stats.iact_phi,    digits=3),
                    iact_loglambda = round.(stats.iact_lambda, digits=3),
                )
            end
        end
    end
end
