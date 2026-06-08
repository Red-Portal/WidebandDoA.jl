
using AbstractMCMC
using Accessors
using Base.GC
using DataFrames
using Distributed
using Distributions
using DrWatson
using HDF5
using MCMCDiagnosticTools
using OnlineStats
using ProgressMeter
using Random, Random123
using ReversibleJump
using WidebandDoA
using mcmcse

@quickactivate

include("common.jl")

function run_rjmcmc(rng, cond, n_samples, rjmcmc)
    initial_params = WidebandDoA.WidebandIsoIsoParam{Float64}[]
    initial_order  = 0
    _, stats = ReversibleJump.sample(
        rng,
        rjmcmc,
        cond,
        n_samples,
        initial_order,
        initial_params;
        show_progress=false,
    )
    return stats
end

function estimate_true_posterior(rng, cond)
    n_burn    = 2^16
    n_samples = 2^16

    prop   = UniformNormalLocalProposal(0.0, 1.0)
    mcmc   = SliceSteppingOut([2.0, 2.0])
    jump   = IndepJumpProposal(prop)
    rjmcmc = ReversibleJump.NonReversibleJumpMCMC(jump, mcmc; jump_rate=0.9)

    stats = run_rjmcmc(rng, cond, n_samples + n_burn, rjmcmc)
    stats = stats[n_burn+1:end]
    return modelposterior_naive(stats), var([stat.order for stat in stats])
end

function total_variation_distance(p, q)
    @assert p.support == q.support
    return sum(abs, p.p - q.p)/2
end

function run_simulation(n_targets, snr, rjmcmc, n_burn, n_samples, n_reps)
    stats = @showprogress pmap(1:n_reps) do key
        seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
        rng  = Philox4x(UInt64, seed, 8)
        set_counter!(rng, key)

	N       = 128
        n_dft   = 1024
        fs      = 3000.0
	model   = construct_default_model(N, fs)
        f_begin = 10
        f_end   = 1000

        ϕ     = rand(rng, Uniform(-π/2, π/2), n_targets)
	c, Δx = model.likelihood.c, model.likelihood.Δx
	y, _  = simulate_signal(rng, N, n_dft, ϕ, snr, f_begin, f_end, fs, 1.0, Δx, c)
	cond  = WidebandConditioned(model, y)

        k_post_true, k_var_true = estimate_true_posterior(rng, cond)

        stats = run_rjmcmc(rng, cond, n_samples + n_burn, rjmcmc)

        # Compute convergence in TV distance
        sup      = k_post_true.support
        hist     = Hist(0:maximum(sup) + 1)
        tv_curve = map(stats) do stat
            fit!(hist, stat.order)
            _, counts = value(hist)
            prob        = counts/sum(counts)
            k_post      = DiscreteNonParametric(sup, prob)
            total_variation_distance(k_post, k_post_true)
        end

        # Integrated autocorrelation time
        k_chain = [stat.order for stat in stats[n_burn+1:end]]
        iact    = if k_var_true < eps(Float32)
            0
        else
            mcmcse.mcvar(k_chain, r=2)/k_var_true
        end
        (tv_curve = tv_curve, iact = iact, k_chain = k_chain)
    end
end

function main()
    n_burn    = 2^10
    n_samples = 2^17
    n_reps    = 32

    prop = UniformNormalLocalProposal(0.0, 2.0)
    mcmc = SliceSteppingOut([2.0, 2.0])
    jump = IndepJumpProposal(prop)
    dfs  = DataFrame()
    for (mcmc, mcmc_name) in [
        (
            WidebandIsoIsoMetropolis(
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
            SliceSteppingOut([2.0, 2.0]),
            "slice"
        )
    ], rjmcmc_name in ["rjmcmc", "nrjmcmc"]

        rjmcmc = if rjmcmc_name == "rjmcmc"
            order_prior = truncated(NegativeBinomial(1/2 + 0.1, 0.1/(0.1 + 1)), 0, 20)
            ReversibleJump.ReversibleJumpMCMC(order_prior, jump, mcmc)
        else
            ReversibleJump.NonReversibleJumpMCMC(jump, mcmc; jump_rate=0.9)
        end

        for snr in [-8, -4, 0, 4], n_targets in [2, 8]
            @info("simualting", rjmcmc=rjmcmc_name, mcmc=mcmc_name, snr, n_targets)

            stats = run_simulation(n_targets, snr, rjmcmc, n_burn, n_samples, n_reps)
            df = DataFrame(
                rjmcmc    = rjmcmc_name,
                mcmc      = mcmc_name,
                snr       = snr,
                n_targets = n_targets,
                tv_curve  = [stat.tv_curve for stat in stats],
                iact      = [stat.iact     for stat in stats],
            )
            GC.gc()
            dfs = vcat(dfs, df)
            save(datadir("raw", "rjmcmc_eval.jld2"), "data", dfs) 
        end
    end
    #reduce_namedtuples(
    #    arr -> run_bootstrap(arr; confint_strategy=BCaConfInt(0.95)), stats
    #)
end
