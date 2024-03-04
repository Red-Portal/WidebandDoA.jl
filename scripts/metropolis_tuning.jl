
using AbstractMCMC
using Accessors
using Distributions
using DrWatson
using HDF5
using MCMCDiagnosticTools
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

function estimate_acceptance_rate(rng, mcmc, target, θ0, n_burn::Int, n_samples::Int)
    θ           = copy(θ0)
    stats_chain = Vector{NamedTuple}(undef, n_samples)

    @showprogress for _ in 1:n_burn
        θ, _, _ = transition_mcmc(rng, mcmc, target, θ)
    end

    @showprogress for t in 1:n_samples
        θ, _, stats    = transition_mcmc(rng, mcmc, target, θ)
        stats_chain[t] = stats
    end
    reduce_namedtuples(mean, stats_chain)
end

function run_simulation(snr, mcmc, n_burn, n_samples, n_reps)
    stats = map(1:n_reps) do key
        seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
        rng  = Philox4x(UInt64, seed, 8)
        set_counter!(rng, key)

        ϕ     = rand(rng, Uniform(-π/2, π/2))
        model = construct_default_model(rng, [ϕ], snr)
        θ0    = [WidebandDoA.WidebandNormalGammaParam(ϕ, randn(rng))]
        estimate_acceptance_rate(rng, mcmc, model, θ0, n_burn, n_samples)
    end
    reduce_namedtuples(
        arr -> run_bootstrap(arr; confint_strategy=NormalConfInt(0.95)), stats
    )
end

function main()
    n_burn    = 128
    n_samples = 512
    n_reps    = 32

    for mcmc in [
        WidebandNormalGammaMetropolis(
            IndependentMetropolis(Uniform(-π/2, π/2)),
            IndependentMetropolis(Normal(0, 2))
        ),
        WidebandNormalGammaMetropolis(
            RandomWalkMetropolis(0.1),
            RandomWalkMetropolis(1.5)
        )
        ]
        for snr in [-8, -4, 0, 4]
            (; phi_acceptance_rate, loglambda_acceptance_rate) = run_simulation(
                snr, mcmc, n_burn, n_samples, n_reps
            )
            @info(
                "",
                mcmc,
                snr,
                phi_acceptance_rate       = round.(phi_acceptance_rate,       digits=3),
                loglambda_acceptance_rate = round.(loglambda_acceptance_rate, digits=3),
            )
        end
    end
end
